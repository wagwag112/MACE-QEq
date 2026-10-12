from pathlib import Path

ROOT = Path("/scratch/tmp/hdang/projects/Li_Li6PS5Cl")
PROJECT_ROOT = ROOT

ROUND = 0
PARENT_ROUND = -1
ANODE = "Li"
SE = "Li6PS5Cl"
SYSTEM = f"{ANODE}_{SE}"

ACTIVE_METAL = ANODE
RCUT_METAL = 3.50
MIN_CLUSTER_SIZE = 8

SE_ELEMENT_CUTOFFS = {
    "N": 2.50,
    "P": 2.80,
    "S": 2.90,
    "Cl": 3.05,
}
ANION_CUTOFFS = SE_ELEMENT_CUTOFFS
NON_METAL_CUTOFFS = SE_ELEMENT_CUTOFFS

CHI_HARDNESS_TABLE = {
    "Li": (-3.0000, 10.0241),
    "N": (7.6100, 7.7500),
    "P": (1.8000, 7.0946),
    "S": (6.5745, 9.0000),
    "Cl": (9.9704, 6.1703),
}
EXPLORATION_MODE = "constp"
BOTTOM_EXTERNAL_CHI = 6.0
UPPER_EXTERNAL_CHI = -2.0
TEMPERATURE_K = 300
MD_PS = 50
TIMESTEP_FS = 2.0
SAVE_EVERY = 10
MD_SEED = 20261004
TTIME_FS = 20.0

PAIR_CAPACITY_FACTOR = 1.1
MAX_PAIRS = 0
DEVICE = "cuda"

N_HIGH = 65
N_COVERAGE = 15
N_HOLDOUT = 20
N_STANDARD_ENRICH = N_HIGH + N_COVERAGE
N_ENRICH = N_STANDARD_ENRICH
N_SELECTED = N_ENRICH + N_HOLDOUT

HIGH_MIN_SPACING_PS = round(MD_PS / 140.0, 3)
MIN_HOLDOUT_DISTANCE_PS = round(MD_PS / 500.0, 3)
COVERAGE_MIN_SPACING_PS = round(MD_PS / 1500.0, 3)
EXPECTED_VALID = 100

PROBE_MD_PS = 50
PROBE_TEMPERATURE_K = 300
PROBE_MODE = "constp"
PROBE_BOTTOM_EXTERNAL_CHI = BOTTOM_EXTERNAL_CHI
PROBE_UPPER_EXTERNAL_CHI = UPPER_EXTERNAL_CHI
PROBE_MD_SEED = MD_SEED + 1000 + ROUND

CONV_SIGMA_P95_TARGET = 0.100
CONV_SIGMA_P95_HARD = 0.150
CONV_MAX_FRAC_EXCEED_TARGET = 0.01
CONV_MAX_FRAC_EXCEED_HARD = 0.00
CONV_MAX_TEST_FORCE_RMSE_REGRESSION = 0.10
CONV_MAX_HOLDOUT_FORCE_RMSE_REGRESSION = 0.20

COMMITTEE_SEEDS = [20261004, 20261005, 20261006, 20261007]
BASE_SEED = COMMITTEE_SEEDS[0]

COVERAGE_WINDOWS = [
    (round(i * (MD_PS / N_COVERAGE), 4), round((i + 1) * (MD_PS / N_COVERAGE), 4))
    for i in range(N_COVERAGE)
]
HOLDOUT_WINDOWS = [
    (round(i * (MD_PS / N_HOLDOUT), 4), round((i + 1) * (MD_PS / N_HOLDOUT), 4))
    for i in range(N_HOLDOUT)
]

TOTAL_MD_STEPS = int(round(MD_PS * 1000.0 / TIMESTEP_FS))
EXPECTED_MD_FRAMES = TOTAL_MD_STEPS // SAVE_EVERY + 1
FRAME_INTERVAL_PS = SAVE_EVERY * TIMESTEP_FS / 1000.0

PROBE_TOTAL_MD_STEPS = int(round(PROBE_MD_PS * 1000.0 / TIMESTEP_FS))
PROBE_EXPECTED_MD_FRAMES = PROBE_TOTAL_MD_STEPS // SAVE_EVERY + 1

MD_ROOT = ROOT / "md_validation"
DATA_DIR = ROOT / "data"
ACTIVE_LEARNING_DIR = ROOT / "active_learning"




CONSTQ_INPUT_STRUCTURE = PROJECT_ROOT / "structures" / f"{SYSTEM}_test.vasp"
CONSTP_INPUT_STRUCTURE = PROJECT_ROOT / "structures" / f"{SYSTEM}_tv.vasp"
INPUT_STRUCTURE = CONSTP_INPUT_STRUCTURE if EXPLORATION_MODE == "constp" else CONSTQ_INPUT_STRUCTURE
PROBE_INPUT_STRUCTURE = CONSTP_INPUT_STRUCTURE if PROBE_MODE == "constp" else CONSTQ_INPUT_STRUCTURE

PARENT_TRAIN_DFT = (DATA_DIR / f"train_v{PARENT_ROUND}_dft.extxyz") if (PARENT_ROUND is not None and PARENT_ROUND >= 0) else None
PARENT_TRAIN_SHORT = (DATA_DIR / f"train_v{PARENT_ROUND}_short.extxyz") if (PARENT_ROUND is not None and PARENT_ROUND >= 0) else None
PARENT_VALID_DFT = DATA_DIR / "valid_dft.extxyz"
PARENT_VALID_SHORT = DATA_DIR / "valid_short.extxyz"

TRAINED_MODELS_ROOT = ROOT / "trained_models"

PRETRAINED_MACE_MODEL = TRAINED_MODELS_ROOT / "mace-mpa-0-medium.model"

V0_BOOTSTRAP_DIR = MD_ROOT / "v0_bootstrap"
V0_TRAJ_DIR = V0_BOOTSTRAP_DIR / "trajectories"
V0_SAMPLED_DIR = V0_BOOTSTRAP_DIR / "sampled_extxyz"
V0_TARGET_COUNT = 400

V0_SPECS = {
    "test": {
        ANODE: {"n_frames": 10, "dt_ps": 1.0, "t_equil_ps": 2.0, "struct": PROJECT_ROOT / "structures" / f"{ANODE}_test.vasp"},
        SE: {"n_frames": 10, "dt_ps": 1.0, "t_equil_ps": 2.0, "struct": PROJECT_ROOT / "structures" / f"{SE}_test.vasp"},
        SYSTEM: {"n_frames": 80, "dt_ps": 0.5, "t_equil_ps": 2.0, "struct": PROJECT_ROOT / "structures" / f"{SYSTEM}_test.vasp"},
    },
    "tv": {
        ANODE: {"n_train": 50, "n_valid": 10, "dt_ps": 1.0, "t_equil_ps": 2.0, "struct": PROJECT_ROOT / "structures" / f"{ANODE}_tv.vasp"},
        SE: {"n_train": 50, "n_valid": 10, "dt_ps": 1.0, "t_equil_ps": 2.0, "struct": PROJECT_ROOT / "structures" / f"{SE}_tv.vasp"},
        SYSTEM: {"n_train": 100, "n_valid": 80, "dt_ps": 0.5, "t_equil_ps": 2.0, "struct": PROJECT_ROOT / "structures" / f"{SYSTEM}_tv.vasp"},
    },
}

if ROUND == 0:
    CURRENT_FULL_MODEL = PRETRAINED_MACE_MODEL
    CURRENT_SHORT_MODEL = PRETRAINED_MACE_MODEL
else:
    CURRENT_FULL_MODEL = TRAINED_MODELS_ROOT / f"train_full_v{PARENT_ROUND}" / "checkpoints" / f"MACE_full_v{PARENT_ROUND}_run-{BASE_SEED}.model"
    CURRENT_SHORT_MODEL = TRAINED_MODELS_ROOT / f"train_short_v{PARENT_ROUND}" / "checkpoints" / f"MACE_short_v{PARENT_ROUND}_run-{BASE_SEED}.model"

COMMITTEE_ROOT = TRAINED_MODELS_ROOT / f"committee_short_v{PARENT_ROUND}"
BASE_COMMITTEE_MODEL = CURRENT_SHORT_MODEL

ROUND_DIR = MD_ROOT / f"active_learning_v{ROUND}"
EXPLORATION_DIR = ROUND_DIR / "exploration"
SCORING_DIR = ROUND_DIR / "committee_scoring"
FRAMES_DIR = ROUND_DIR / "frames_extracted"
DFT_DIR = MD_ROOT / f"v{ROUND}"

MD_RUN_NAME = f"{SYSTEM}_short_v{PARENT_ROUND}_{TEMPERATURE_K}K_{MD_PS:g}ps"
MD_OUTPUT_DIR = EXPLORATION_DIR / MD_RUN_NAME
TRAJECTORY_FILE = MD_OUTPUT_DIR / "md.extxyz"
TRAJECTORY_ASE_FILE = MD_OUTPUT_DIR / "md.traj"
THERMO_FILE = MD_OUTPUT_DIR / "thermo.csv"

PREV_PROBE_DIR = (MD_ROOT / f"active_learning_v{PARENT_ROUND}" / "validation" / "fresh_probe" / f"{SYSTEM}_short_v{PARENT_ROUND}_{PROBE_TEMPERATURE_K}K_{PROBE_MD_PS:g}ps_probe") if (PARENT_ROUND is not None and PARENT_ROUND >= 0) else None
PREV_PROBE_TRAJ = (PREV_PROBE_DIR / "md.extxyz") if PREV_PROBE_DIR is not None else None
SELECTION_TRAJECTORY_FILE = PREV_PROBE_TRAJ if (PARENT_ROUND is not None and PARENT_ROUND >= 0) else TRAJECTORY_FILE

SELECTION_SCORE_FILE = SCORING_DIR / "committee_scores.csv"
SCORE_FILE = SCORING_DIR / "committee_scores.csv"
TOP_UNCERTAINTY_FILE = SCORING_DIR / "top50_uncertainty.csv"
SCORING_FIGURE_DIR = SCORING_DIR / "figures"
SELECTION_MANIFEST = FRAMES_DIR / "selection_manifest.csv"
DFT_SELECTED_DIR = ROUND_DIR / "dft_selected"
DFT_BATCH_NAME = f"{SYSTEM}_train_v{ROUND}"


DFT_FILE = DFT_DIR / f"md_v{ROUND}_dft.extxyz"
DFT_REPORT = DFT_DIR / f"md_v{ROUND}_dft_report.csv"
SHORT_FILE = DFT_DIR / f"md_v{ROUND}_short.extxyz"
QEQ_REPORT = DFT_DIR / f"md_v{ROUND}_qeq_report.csv"

ROUND_TRAIN_DFT = DFT_DIR / f"v{ROUND}_train_dft.extxyz"
ROUND_TRAIN_SHORT = DFT_DIR / f"v{ROUND}_train_short.extxyz"
ROUND_HOLDOUT_DFT = DFT_DIR / f"v{ROUND}_holdout_dft.extxyz"
ROUND_HOLDOUT_SHORT = DFT_DIR / f"v{ROUND}_holdout_short.extxyz"

OUT_TRAIN_DFT = DATA_DIR / f"train_v{ROUND}_dft.extxyz"
OUT_TRAIN_SHORT = DATA_DIR / f"train_v{ROUND}_short.extxyz"
OUT_VALID_DFT = DATA_DIR / "valid_dft.extxyz"
OUT_VALID_SHORT = DATA_DIR / "valid_short.extxyz"

TRAIN_FULL_DIR = TRAINED_MODELS_ROOT / f"train_full_v{ROUND}"
TRAIN_SHORT_DIR = TRAINED_MODELS_ROOT / f"train_short_v{ROUND}"
COMMITTEE_NEXT_DIR = TRAINED_MODELS_ROOT / f"committee_short_v{ROUND}"

MODEL_FULL_NAME = f"MACE_full_v{ROUND}"
MODEL_SHORT_NAME = f"MACE_short_v{ROUND}"

VALIDATION_DIR = MD_ROOT / f"active_learning_v{ROUND}" / "validation"
EVALUATION_DIR = VALIDATION_DIR / "evaluation"
FRESH_PROBE_DIR = VALIDATION_DIR / "fresh_probe"
FRESH_PROBE_SCORING_DIR = VALIDATION_DIR / "fresh_probe_scoring"
CONVERGENCE_DIR = VALIDATION_DIR / "convergence"

CURRENT_GENERATION_FULL_MODEL = TRAIN_FULL_DIR / "checkpoints" / f"{MODEL_FULL_NAME}_run-{BASE_SEED}.model"
CURRENT_GENERATION_SHORT_MODEL = TRAIN_SHORT_DIR / "checkpoints" / f"{MODEL_SHORT_NAME}_run-{BASE_SEED}.model"

EVALUATION_SUMMARY = EVALUATION_DIR / (f"v{ROUND}_standalone_summary.csv" if (PARENT_ROUND is None or PARENT_ROUND < 0) else f"v{PARENT_ROUND}_vs_v{ROUND}_summary.csv")
EVALUATION_FRAMES = EVALUATION_DIR / (f"v{ROUND}_standalone_frames.csv" if (PARENT_ROUND is None or PARENT_ROUND < 0) else f"v{PARENT_ROUND}_vs_v{ROUND}_frames.csv")

PROBE_RUN_NAME = f"{SYSTEM}_short_v{ROUND}_{PROBE_TEMPERATURE_K}K_{PROBE_MD_PS:g}ps_probe"
PROBE_OUTPUT_DIR = FRESH_PROBE_DIR / PROBE_RUN_NAME
PROBE_TRAJECTORY_FILE = PROBE_OUTPUT_DIR / "md.extxyz"
PROBE_TRAJECTORY_ASE_FILE = PROBE_OUTPUT_DIR / "md.traj"
PROBE_THERMO_FILE = PROBE_OUTPUT_DIR / "thermo.csv"

PROBE_SCORE_FILE = FRESH_PROBE_SCORING_DIR / "committee_scores.csv"
PROBE_TOP_UNCERTAINTY_FILE = FRESH_PROBE_SCORING_DIR / "top50_uncertainty.csv"
PROBE_SCORING_FIGURE_DIR = FRESH_PROBE_SCORING_DIR / "figures"

CONVERGENCE_REPORT = CONVERGENCE_DIR / "convergence_report.txt"
CONVERGENCE_JSON = CONVERGENCE_DIR / "convergence_report.json"


def validate_config():
    if ROUND == 0:
        if PARENT_ROUND is not None and PARENT_ROUND != -1:
            raise RuntimeError("PARENT_ROUND must be None or -1 for ROUND 0.")
        total_v0 = sum(p["n_frames"] for p in V0_SPECS["test"].values()) + sum(p["n_train"] + p["n_valid"] for p in V0_SPECS["tv"].values())
        if total_v0 != V0_TARGET_COUNT:
            raise RuntimeError(f"V0 target frame mismatch: {total_v0} != {V0_TARGET_COUNT}")
    else:
        if PARENT_ROUND is None or ROUND <= PARENT_ROUND or ROUND != PARENT_ROUND + 1:
            raise RuntimeError("ROUND must equal PARENT_ROUND + 1.")
    if not SYSTEM:
        raise RuntimeError("SYSTEM cannot be empty.")
    if not ACTIVE_METAL:
        raise RuntimeError("ACTIVE_METAL cannot be empty.")
    if RCUT_METAL <= 0.0:
        raise RuntimeError("RCUT_METAL must be positive.")
    if MIN_CLUSTER_SIZE < 1:
        raise RuntimeError("MIN_CLUSTER_SIZE must be positive.")
    if not isinstance(ANION_CUTOFFS, dict) or not ANION_CUTOFFS:
        raise RuntimeError("ANION_CUTOFFS must be a non-empty dictionary.")
    for sym, rcut in ANION_CUTOFFS.items():
        if not isinstance(sym, str) or rcut <= 0.0:
            raise RuntimeError(f"Invalid anion cutoff: {sym}: {rcut}")
    if EXPLORATION_MODE not in {"constq", "constp"}:
        raise RuntimeError("EXPLORATION_MODE must be 'constq' or 'constp'.")
    if not isinstance(BOTTOM_EXTERNAL_CHI, (int, float)):
        raise RuntimeError("BOTTOM_EXTERNAL_CHI must be numeric.")
    if not isinstance(UPPER_EXTERNAL_CHI, (int, float)):
        raise RuntimeError("UPPER_EXTERNAL_CHI must be numeric.")
    if EXPECTED_VALID <= 0:
        raise RuntimeError("EXPECTED_VALID must be positive.")
    if MD_PS <= 0.0:
        raise RuntimeError("MD_PS must be positive.")
    if TIMESTEP_FS <= 0.0:
        raise RuntimeError("TIMESTEP_FS must be positive.")
    if TTIME_FS <= 0.0:
        raise RuntimeError("TTIME_FS must be positive.")
    if SAVE_EVERY <= 0:
        raise RuntimeError("SAVE_EVERY must be positive.")
    if N_HIGH < 0 or N_COVERAGE < 0 or N_HOLDOUT < 1:
        raise RuntimeError("Invalid candidate selection quotas.")
    if N_HIGH + N_COVERAGE + N_HOLDOUT != N_SELECTED:
        raise RuntimeError("Selection quota mismatch.")
    if HIGH_MIN_SPACING_PS <= 0.0 or MIN_HOLDOUT_DISTANCE_PS <= 0.0:
        raise RuntimeError("Spacing parameters must be positive.")
    if len(COVERAGE_WINDOWS) != N_COVERAGE or len(HOLDOUT_WINDOWS) != N_HOLDOUT:
        raise RuntimeError("Window length mismatch.")
    if len(COMMITTEE_SEEDS) < 2:
        raise RuntimeError("Committee must contain at least 2 models.")
    if TOTAL_MD_STEPS % SAVE_EVERY != 0:
        raise RuntimeError("TOTAL_MD_STEPS must be divisible by SAVE_EVERY.")
    if PROBE_MODE not in {"constq", "constp"}:
        raise RuntimeError("PROBE_MODE must be 'constq' or 'constp'.")
    if PROBE_MD_PS <= 0.0:
        raise RuntimeError("PROBE_MD_PS must be positive.")
    if PROBE_TOTAL_MD_STEPS % SAVE_EVERY != 0:
        raise RuntimeError("PROBE_TOTAL_MD_STEPS must be divisible by SAVE_EVERY.")

validate_config()

DFT_PARTITIONS = [
    "normal", "long", "requeue", "requeue-zen", "zen4",
    "himszen4", "hims", "requeue-gpu", "q0heuer"
]
DFT_CPUS_PER_JOB = 24
DFT_TIME_LIMIT = "12:00:00"

DFT_ROOT = ROOT / "DFT"
DFT_CALC_DIR = DFT_ROOT / f"v{ROUND}" / "cal"
DFT_TEMPLATE_DIR = ACTIVE_LEARNING_DIR / "template"
DFT_RUN_SCRIPT = ACTIVE_LEARNING_DIR / "slurm" / "run_dft.sh"

