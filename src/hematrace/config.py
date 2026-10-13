"""Original research configuration and run-specific output directories."""

import time
import warnings
from dataclasses import dataclass
from pathlib import Path

warnings.filterwarnings("ignore")

@dataclass
class CFG:
    PROJECT_NAME: str = "HemaTRACE_MANUSCRIPT"
    DATA_ROOT: str = "Dataset"
    MIMIC_SUBDIR: str = "Mimic4"
    OUT_ROOT: str = "HemaTrace_results"

    INPUT_START_HOURS: float = 0.0
    INPUT_END_HOURS: float = 24.0
    TARGET_START_HOURS: float = 24.0
    TARGET_END_HOURS: float = 72.0
    BIN_SIZE_HOURS: float = 4.0
    N_BINS: int = 6

    TRAIN_FRAC: float = 0.70
    VAL_FRAC: float = 0.10
    TEST_FRAC: float = 0.20

    SEVERE_HB_THRESHOLD: float = 8.0
    MALE_ANEMIA_THRESHOLD: float = 13.0
    FEMALE_ANEMIA_THRESHOLD: float = 12.0

    CHUNKSIZE_LABEVENTS: int = 2_000_000
    CHUNKSIZE_CHARTEVENTS: int = 2_000_000
    CHUNKSIZE_INPUTEVENTS: int = 1_000_000

    RANDOM_STATE: int = 42

    # Neural model
    MAX_EPOCHS: int = 80
    PATIENCE: int = 12
    BATCH_SIZE: int = 512
    NUM_WORKERS: int = 12
    LR: float = 8e-4
    WEIGHT_DECAY: float = 1e-4
    GRAD_CLIP: float = 1.0
    USE_AMP: bool = True

    TEMPORAL_EMBED_DIM: int = 128
    GRU_HIDDEN_DIM: int = 128
    GRU_LAYERS: int = 1
    GRU_BIDIRECTIONAL: bool = False
    STATIC_HIDDEN_DIM: int = 128
    STATIC_EMBED_DIM: int = 64
    SHARED_HIDDEN_DIM: int = 192
    DROPOUT: float = 0.30

    REG_LOSS_WEIGHT: float = 1.0
    SEVERE_LOSS_WEIGHT: float = 0.7
    ONSET_LOSS_WEIGHT: float = 0.5

    SCORE_ONSET_WEIGHT: float = 0.30
    SCORE_MAE_WEIGHT: float = 0.60

    ONSET_FINETUNE_EPOCHS: int = 35
    ONSET_FINETUNE_PATIENCE: int = 8
    ONSET_FINETUNE_LR: float = 3e-4

    THRESHOLD_GRID_MIN: float = 0.05
    THRESHOLD_GRID_MAX: float = 0.95
    THRESHOLD_GRID_STEP: float = 0.01

    BOOTSTRAP_N: int = 1000
    BOOTSTRAP_SEED: int = 123

    XAI_MAX_SAMPLES: int = 2000
    XAI_BATCH_SIZE: int = 512
    PERM_IMPORTANCE_N: int = 1500

    DPI: int = 300

    FAST_DEBUG: bool = False
    FAST_DEBUG_MAX_SUBJECTS: int = 3000
    FAST_DEBUG_MAX_CHUNKS: int = 2
    FAST_DEBUG_BOOTSTRAP_N: int = 50


cfg = CFG()

RUN_DIR = Path(cfg.OUT_ROOT) / cfg.PROJECT_NAME / time.strftime("T_run_%Y-%m-%d_at_%H-%M-%S")
DATA_DIR = RUN_DIR / "data"
MODELS_DIR = RUN_DIR / "models"
TABULAR_MODEL_DIR = MODELS_DIR / "tabular"
TEMPORAL_MODEL_DIR = MODELS_DIR / "temporal"
ONSET_MODEL_DIR = MODELS_DIR / "onset_specialized"
FUSION_MODEL_DIR = MODELS_DIR / "fusion"
TABLES_DIR = RUN_DIR / "tables"
FIGURES_DIR = RUN_DIR / "figures"
FIGURE_PNG_DIR = FIGURES_DIR / "png"
FIGURE_PDF_DIR = FIGURES_DIR / "pdf"
XAI_DIR = RUN_DIR / "xai"
REPORTS_DIR = RUN_DIR / "reports"
LOGS_DIR = RUN_DIR / "logs"
TMP_DIR = RUN_DIR / "tmp_filtered_events"

for d in [DATA_DIR, MODELS_DIR, TABULAR_MODEL_DIR, TEMPORAL_MODEL_DIR, ONSET_MODEL_DIR,
          FUSION_MODEL_DIR, TABLES_DIR, FIGURES_DIR, FIGURE_PNG_DIR, FIGURE_PDF_DIR,
          XAI_DIR, REPORTS_DIR, LOGS_DIR, TMP_DIR]:
    d.mkdir(parents=True, exist_ok=True)
