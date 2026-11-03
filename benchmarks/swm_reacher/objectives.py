"""Planning objectives for the harness's ``objective=`` slot."""
from torch import nn


class GoalMSEEveryStep(nn.Module):
    """``GoalMSE`` summed over every predicted step, not only the last: rewards arriving early and staying."""

    def forward(self, info_dict):
        pred = info_dict["predicted_emb"]                       # (B, S, H + T, D)
        goal = info_dict["goal_emb"][:, None, -1:]              # (B, 1, 1, D)
        n_ctx = info_dict["emb"].shape[2]
        return ((pred[:, :, n_ctx:] - goal) ** 2).sum((2, 3))
