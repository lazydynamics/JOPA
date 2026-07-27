"""Central configuration: numerical guards, model defaults, training presets.

Three kinds of number live here and are used differently.

* **Numerical hygiene** — epsilons, clips and ridges whose only job is to keep
  linear algebra and gradients finite. They must agree everywhere they appear,
  so call sites reference them directly rather than re-deriving a literal.
* **Model and algorithm defaults** — priors, precisions, iteration counts,
  neighbourhood sizes, network shapes. These stay keyword arguments of the
  classes and functions that use them; only the literal is replaced by the name
  below, so every call site remains discoverable and overridable.
* **Training hyperparameters** — loss weights, learning rates, step counts.
  These are per-experiment rather than library-wide, so each trainer owns a
  frozen dataclass whose field defaults are the values the recorded experiments
  were produced with.

Values are reproduced exactly as the library shipped them; nothing here was
retuned. Where the rationale for a value could not be recovered from the code or
its history, the comment says so instead of inventing one.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

# ---------------------------------------------------------------------------
# Numerical hygiene
# ---------------------------------------------------------------------------

# A variance/second-moment at or below this is numerically zero: used before a
# sqrt so the gradient stays finite, and to detect a degenerate R² denominator.
EPS_VARIANCE = 1e-12

# Generic denominator floor for ratios of non-negative quantities (mean pixel
# weight, mean diagonal precision, calibration scale). Small enough to be
# inert on well-conditioned data, large enough to stop float32 division blowing up.
EPS_DIVISION = 1e-8

# Added inside a sqrt of a sum of squares so the derivative at exactly zero
# distance is finite (identical pose pairs occur in every batch).
EPS_SQRT = 1e-8


# Floor on a batch-mean normaliser used to make two distance measures
# commensurate before comparing them.
EPS_NORMALIZER = 1e-6


# Encoder log-standard-deviation clip. exp(-6) ≈ 2.5e-3 bounds the reported
# precision (a sensor claiming more is over-confident and destabilises
# filtering); exp(2) ≈ 7.4 bounds it below the latent working scale.
LOG_STD_CLIP = (-6.0, 2.0)

# Bernoulli probability clip before log() in a reconstruction cross-entropy.
PROB_CLIP = (1e-6, 1.0 - 1e-6)

# Per-dimension standard-deviation floor used to whiten replay states before
# nearest-neighbour selection, so a frozen channel cannot dominate the metric.
WHITENING_STD_FLOOR = 1e-3




# ---------------------------------------------------------------------------
# Conjugate priors and initial posteriors for the CT-factor VMP
# ---------------------------------------------------------------------------

PRIOR_W_DF: float = 4.0       # Wishart prior df on the dynamics noise precision
PRIOR_A_COV: float = 1.0      # vec(A) prior variance
PRIOR_B_COV: float = 1.0      # vec(B) prior variance
INIT_A_COV: float = 100.0     # initial q(A) variance — wide so data dominates
INIT_B_COV: float = 100.0     # initial q(B) variance

# A Wishart on a d×d precision needs df > d + 1 for a finite mean; `d + this` is
# the smallest safe proper prior and is used wherever a df floor is required.
WISHART_DF_MARGIN = 2



# ---------------------------------------------------------------------------
# Transition learning: iteration counts and observed-fit precisions
# ---------------------------------------------------------------------------

# Coordinate-ascent VMP sweeps. The updates are conjugate closed forms, so these
# are "enough to converge on the trajectory lengths used here" rather than tuned.
VMP_ITERATIONS = 50               # LearnedLinear (chain smoothing in the loop)

LEARN_OBSERVED_PRECISION = 1e4    # learn_observed default



# ---------------------------------------------------------------------------
# Local replay refits
# ---------------------------------------------------------------------------

# Neighbourhood size for a local conjugate refit: large enough that the
# rectangular fit is well determined, small enough that the neighbourhood is
# still local in whitened state space.
REPLAY_NEIGHBORS = 256



# ---------------------------------------------------------------------------
# Delay-embedding dynamics
# ---------------------------------------------------------------------------

# Frames in a delay state z = [h(t-K+1) … h(t)]. Four steps of history make
# acceleration observable in a second-order system.
DEFAULT_DELAY = 4

# Frames in a pixel-sensor window. The same history length as DEFAULT_DELAY,
# named separately because it is a sensor input shape rather than a state layout.
DEFAULT_WINDOW_FRAMES = DEFAULT_DELAY






# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

# Sweeps for the iterative (mean-field) planner. Unused by method="exact".
PLAN_VMP_ITERATIONS = 50

# Covariance pinning q(A)/q(B) to a point estimate when planning through a
# KnownPhysics block: the linearisation is treated as certain.
POINT_ESTIMATE_COV = 1e-8

# Precision pinning the appended constant-offset control channel to exactly 1.
# JointModel.plan and Agent use DIFFERENT values for the same pin and neither
# origin is recorded; they are kept apart here so the discrepancy is visible
# rather than silently unified.
PLAN_OFFSET_PIN_PRECISION = 1e8
AGENT_OFFSET_PIN_PRECISION = 1e6



# ---------------------------------------------------------------------------
# Cross-block coupling
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Closed-loop agent
# ---------------------------------------------------------------------------








# ---------------------------------------------------------------------------
# Pixel sensors: geometry and message precision
# ---------------------------------------------------------------------------

# Frame sizes the conv trunks have stages for. 28 is the MNIST-era default kept
# for checkpoints written before `img_size` existed.
IMG_SIZE_SMALL = 28
IMG_SIZE_MEDIUM = 64
IMG_SIZE_LARGE = 128
SUPPORTED_IMG_SIZES = (IMG_SIZE_SMALL, IMG_SIZE_MEDIUM, IMG_SIZE_LARGE)
DEFAULT_IMG_SIZE = IMG_SIZE_SMALL
DEFAULT_N_FRAMES = 1

DEFAULT_LATENT_DIM = 2            # VAE latent width
DEFAULT_POSE_DIM = 4              # PoseMotionVAE pose channels
DEFAULT_MOTION_DIM = 2            # PoseMotionVAE motion channels


# Coverage of the per-dimension calibration: offsets are chosen so this fraction
# of held-out innovations falls inside the reported interval.
CALIBRATION_COVERAGE = 0.90

# Eigenvalue floor as a fraction of the nominal pose precision. The saliency
# weighted Jacobian metric is near rank-deficient, so without a floor whole
# latent directions carry almost no restoring force.
GOAL_EIGENVALUE_FLOOR = 0.5


# ---------------------------------------------------------------------------
# Network shape
# ---------------------------------------------------------------------------

ENCODER_CHANNELS = 32             # first conv width; doubled after the first stage
PIXEL_SENSOR_CHANNELS = 64        # wider trunk used by the 64 px pose/motion sensor
TRUNK_WIDTH = 256                 # dense width shared by every encoder/decoder trunk
MOTION_HIDDEN_WIDTH = 256         # dense width of the pose-delta motion head
CONV_KERNEL = (4, 4)              # kernel of every strided conv / transposed conv
CONV_STRIDE = 2                   # halves (encoder) or doubles (decoder) the resolution


# Bias initialiser for a log-std head: exp(-1) ≈ 0.37 starts the sensor
# moderately uncertain, so the variance head neither saturates the clip nor
# claims precision before the mean head has learned anything.
LOG_STD_BIAS_INIT = -1.0



# ---------------------------------------------------------------------------
# Observation M-step (LearnedVAE)
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

DEFAULT_SEED = 0


# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrainingConfig:
    """Base class for a trainer's hyperparameter preset.

    Field defaults are the values the recorded experiments used. Every field is
    also a keyword argument of the trainer, so a preset can be passed either as
    ``trainer(..., config=preset)`` or expanded with ``**preset.kwargs()``;
    `dataclasses.replace` produces a variant of an existing preset.
    """

    def kwargs(self) -> dict:
        """The preset as trainer keyword arguments."""
        return asdict(self)


@dataclass(frozen=True)
class VAETrainingConfig(TrainingConfig):
    """Hyperparameters of `jopa.nn.vae.train_vae`."""

    latent_dim: int = DEFAULT_LATENT_DIM
    ch: int = ENCODER_CHANNELS
    n_frames: int = DEFAULT_N_FRAMES
    img_size: int = DEFAULT_IMG_SIZE
    epochs: int = 200
    batch_size: int = 64
    lr: float = 1e-3
    beta: float = 1.0             # final KL weight — 1.0 is the plain ELBO
    beta_start: float = 0.1       # initial KL weight; low so the encoder locks on first
    beta_warmup: int = 15         # epochs over which beta_start → beta
    seed: int = DEFAULT_SEED


@dataclass(frozen=True)
class PoseMotionTrainingConfig(TrainingConfig):
    """Hyperparameters of `jopa.nn.pose_motion.train_pose_motion_vae`.

    Phase one trains pose from single frames; phase two adds the disposable
    action-conditioned predictor. Loss weights are empirical and only meaningful
    relative to each other — they were tuned together on the reacher pixels
    cycle and are not derived.
    """

    pose_dim: int = DEFAULT_POSE_DIM
    motion_dim: int = DEFAULT_MOTION_DIM
    n_frames: int = DEFAULT_WINDOW_FRAMES
    rollout: int = 12                       # prediction steps ≈ the planning horizon
    ch: int = PIXEL_SENSOR_CHANNELS
    motion_hidden_dim: int = MOTION_HIDDEN_WIDTH
    img_size: int = IMG_SIZE_MEDIUM
    pose_steps: int = 1500                  # phase-one (pose only) iterations
    controlled_steps: int = 4000            # phase-two (pose + dynamics) iterations
    batch_size: int = 256
    lr: float = 3e-4
    lr_decay_alpha: float = 0.1             # cosine floor as a fraction of lr
    reconstruction_weight: float = 20.0     # dominant term: pixels anchor pose
    foreground_focus: float = 50.0          # extra pixel weight on |frame − background|
    variance_weight: float = 20.0           # per-dimension unit-variance anchor
    covariance_weight: float = 1.0          # off-diagonal decorrelation
    visual_distance_weight: float = 5.0     # pose distances track foreground distances
    prediction_weight: float = 10.0         # scale-normalised latent rollout error
    prediction_nll_weight: float = 0.1      # small: shapes the variance head only
    inverse_weight: float = 2.0             # keeps controllable changes in the latent
    fixed_point_weight: float = 2.0         # rest states must be zero-action fixed points
    zero_motion_weight: float = 10.0        # a repeated frame must read as zero motion
    kl_weight: float = 1e-3                 # weak: anti-collapse is handled explicitly
    beta_nll: float = 0.5                   # stop-gradient β-NLL exponent
    controlled_ramp: int = 250              # iterations over which phase two ramps in
    seed: int = DEFAULT_SEED


# Weight multiplier applied to the decorrelation terms when they are re-applied
# to the full [pose, motion] state in phase two, on top of the same terms
# already charged on pose. Empirical; not derived.
PHASE_TWO_DECORRELATION_SCALE = 0.25


# Diagnostic history points recorded per training phase.
TRAINING_HISTORY_POINTS = 100
