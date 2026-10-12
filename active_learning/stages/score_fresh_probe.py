#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import sys
from ase.io import read

SCRIPT_DIR = Path(__file__).resolve().parent
AL_DIR = SCRIPT_DIR.parent
if str(AL_DIR) not in sys.path:
    sys.path.insert(0, str(AL_DIR))

import al_config as cfg
from stages import score_committee as scorer

def require_file(path: Path) -> None:
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing file: {path}")

def find_committee_model(seed: int) -> Path:
    if seed == cfg.BASE_SEED:
        model = cfg.CURRENT_GENERATION_SHORT_MODEL
        require_file(model)
        return model
    checkpoint_dir = cfg.COMMITTEE_NEXT_DIR / f"seed_{seed}" / "checkpoints"
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory missing: {checkpoint_dir}")
    candidates = sorted(p for p in checkpoint_dir.glob("*.model") if "stagetwo" not in p.name.lower())
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one committee model for seed {seed}, found {len(candidates)}")
    return candidates[0]

def validate_probe() -> int:
    require_file(cfg.PROBE_TRAJECTORY_FILE)
    frames = read(cfg.PROBE_TRAJECTORY_FILE, index=":")
    if abs(len(frames) - cfg.PROBE_EXPECTED_MD_FRAMES) > 2 or len(frames) == 0:
        raise RuntimeError(f"Trajectory incomplete: found {len(frames)}, expected {cfg.PROBE_EXPECTED_MD_FRAMES}")
    return len(frames)

def configure_scorer() -> None:
    scorer.ROUND = cfg.ROUND
    scorer.PARENT_ROUND = cfg.ROUND
    scorer.SYSTEM = cfg.SYSTEM
    scorer.TEMPERATURE_K = cfg.TEMPERATURE_K
    scorer.MD_PS = cfg.PROBE_MD_PS
    scorer.DEVICE = cfg.DEVICE
    scorer.TRAJECTORY_FILE = cfg.PROBE_TRAJECTORY_FILE
    scorer.FRAME_INTERVAL_PS = cfg.FRAME_INTERVAL_PS
    scorer.EXPECTED_MD_FRAMES = cfg.PROBE_EXPECTED_MD_FRAMES
    scorer.BASE_SEED = cfg.BASE_SEED
    scorer.COMMITTEE_SEEDS = list(cfg.COMMITTEE_SEEDS)
    scorer.BASE_COMMITTEE_MODEL = cfg.CURRENT_GENERATION_SHORT_MODEL
    scorer.COMMITTEE_ROOT = cfg.COMMITTEE_NEXT_DIR
    scorer.SCORING_DIR = cfg.FRESH_PROBE_SCORING_DIR
    scorer.SCORING_FIGURE_DIR = cfg.PROBE_SCORING_FIGURE_DIR
    scorer.SCORE_FILE = cfg.PROBE_SCORE_FILE
    scorer.TOP_UNCERTAINTY_FILE = cfg.PROBE_TOP_UNCERTAINTY_FILE

def main() -> None:
    cfg.validate_config()
    validate_probe()
    models = [(s, find_committee_model(s)) for s in cfg.COMMITTEE_SEEDS]
    configure_scorer()
    cfg.FRESH_PROBE_SCORING_DIR.mkdir(parents=True, exist_ok=True)
    cfg.PROBE_SCORING_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    scorer.main()
    require_file(cfg.PROBE_SCORE_FILE)
    require_file(cfg.PROBE_TOP_UNCERTAINTY_FILE)
    print(f"FRESH PROBE SCORING COMPLETE: scores={cfg.PROBE_SCORE_FILE}")

if __name__ == "__main__":
    main()
