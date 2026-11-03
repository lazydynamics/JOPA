"""LeWM baselines for the actuator-fault comparison, as harness ``solver=``s.

All three wrap the harness's ``CEMSolver`` around LeWM and see the same data
stream as JOPA (transitions since the run began, ``history_len = 3``, pooled
over the run's episodes):

* ``OnlineFinetuneCEM`` — generic adaptation: Adam on LeWM's own prediction
  loss, updating predictor and action encoder (``solver=cem_online``);
* ``OnlineGainCEM`` — JOPA's structural prior given to LeWM: a per-actuator
  gain on its action input, fitted by Adam (``solver=cem_gain``);
* ``LaplaceGainCEM`` — the same gain by Laplace / Gauss–Newton, i.e. JOPA's
  inference through LeWM's network (``solver=cem_laplace_gain``).
"""
import torch


class OnlineFinetuneCEM:
    """LeWM + CEM, with LeWM fine-tuned online on the transitions observed so far.

    Same data stream as JOPA (``history_len = 3``,
    pooled over the run's episodes). Before each CEM solve: ``steps`` Adam
    steps on LeWM's own training loss (predict the next embeddings from the
    two previous ones and their action blocks), updating the predictor and the
    action encoder; the image encoder stays frozen.
    """

    def __init__(self, cost, steps=20, lr=1e-4, batch=256, **cem_kwargs):
        from stable_worldmodel.planning.solver import CEMSolver
        self.cem = CEMSolver(cost, **cem_kwargs)
        self.model = cost.model
        self.params = [p for m in (self.model.predictor, self.model.action_encoder, self.model.pred_proj)
                       for p in m.parameters()]
        self.opt = torch.optim.Adam(self.params, lr=lr)
        self.steps, self.batch = steps, batch
        self.emb, self.act = [], []
        self.gen = torch.Generator(device=cem_kwargs.get("device", "cuda")).manual_seed(cem_kwargs.get("seed", 0))

    def configure(self, **kw):
        self.cem.configure(**kw)

    def __getattr__(self, name):                       # n_envs, action_dim, horizon, ...
        return getattr(self.__dict__["cem"], name)

    def __call__(self, *args, **kwargs):
        return self.solve(*args, **kwargs)

    def _finetune(self):
        emb, act = torch.cat(self.emb), torch.cat(self.act)
        for p in self.params:
            p.requires_grad_(True)
        with torch.enable_grad():
            for _ in range(self.steps):
                idx = torch.randint(0, len(emb), (min(self.batch, len(emb)),), device=emb.device, generator=self.gen)
                e, a = emb[idx], act[idx]
                pred = self.model.predict(e[:, :2], self.model.action_encoder(a))
                loss = ((pred - e[:, 1:]) ** 2).mean()
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
        for p in self.params:
            p.requires_grad_(False)
        return loss.item()

    def solve(self, info_dict, init_action=None):
        px = info_dict["pixels"]
        if px.shape[1] >= 3 and "action_history" in info_dict:
            with torch.no_grad():
                e = self.model.encode({"pixels": px.to(self.cem.device)})["emb"].float()
            self.emb.append(e[:, -3:])
            self.act.append(info_dict["action_history"][:, -2:].to(self.cem.device).float())
            loss = self._finetune()
            print(f"finetune loss {loss:.4f} on {sum(len(x) for x in self.emb)} transitions")
        # CEM plans from the current frame only, as in the upstream configuration
        info = dict(info_dict)
        info["pixels"] = px[:, -1:]
        info.pop("action_history", None)
        return self.cem.solve(info, init_action)


class OnlineGainCEM(OnlineFinetuneCEM):
    """LeWM + CEM with the same structural prior as JOPA's gain model: a learned
    per-actuator gain g on LeWM's (raw-unit) action input, fitted online by Adam
    on LeWM's prediction loss; all LeWM weights frozen."""

    def __init__(self, cost, act_mu=(0.0, 0.0), act_sd=(0.5774, 0.5774), **kwargs):
        super().__init__(cost, **kwargs)
        dev = self.cem.device
        self.g = torch.nn.Parameter(torch.ones(2, device=dev))
        mu, sd = torch.tensor(act_mu, device=dev), torch.tensor(act_sd, device=dev)
        self._mu_sd = (mu, sd)
        enc = self.model.action_encoder
        g = self.g

        class GainedEncoder(torch.nn.Module):            # standardised -> raw -> ×g -> standardised
            def __init__(self):
                super().__init__()
                self.enc = enc

            def forward(self, a):
                raw = a.reshape(*a.shape[:-1], -1, 2) * sd + mu
                return self.enc(((raw * g - mu) / sd).reshape(a.shape))

        self.model.action_encoder = GainedEncoder()
        self.params = [self.g]
        self.opt = torch.optim.Adam(self.params, lr=kwargs.get("lr", 1e-2))

    def _finetune(self):
        loss = super()._finetune()
        print(f"gain {self.g.detach().tolist()}")
        return loss


class LaplaceGainCEM(OnlineGainCEM):
    """``OnlineGainCEM`` with JOPA's inference instead of Adam: Laplace / Gauss–Newton
    on the two gains, prior N(1, prior_var), unit observation noise scaled by
    ``noise_var`` (LeWM's one-step residual variance on the training data)."""

    def __init__(self, cost, prior_var=1.0, noise_var=1e-2, gn_steps=5, **kwargs):
        super().__init__(cost, **kwargs)
        self.prior_var, self.noise_var, self.gn_steps = prior_var, noise_var, gn_steps

    def _finetune(self):
        emb, act = torch.cat(self.emb), torch.cat(self.act)
        g0 = self.g.detach().clone()

        def resid(g):
            mu, sd = self._mu_sd
            raw = act.reshape(*act.shape[:-1], -1, 2) * sd + mu
            a = ((raw * g - mu) / sd).reshape(act.shape)
            pred = self.model.predict(emb[:, :2], self.model.action_encoder.enc(a))
            return torch.cat([((pred - emb[:, 1:]) / self.noise_var ** 0.5).flatten(),
                              (g - 1.0) / self.prior_var ** 0.5])

        g = g0
        with torch.no_grad():
            for _ in range(self.gn_steps):
                # central differences: LeWM's attention kernel has no forward-mode AD
                h = 1e-2
                J = torch.stack([(resid(g + h * e) - resid(g - h * e)) / (2 * h) for e in torch.eye(2, device=g.device)], 1)
                g = g - torch.linalg.solve(J.T @ J, J.T @ resid(g))
        self.g.data.copy_(g)
        print(f"gain {g.tolist()}")
        return float((resid(g)[:-2] ** 2).mean())
