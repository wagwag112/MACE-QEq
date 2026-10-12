#!/usr/bin/env python3
from __future__ import annotations
import subprocess
import sys
from pathlib import Path
from ase.io import read

STAGE_DIR = Path(__file__).resolve().parent
AL_DIR = STAGE_DIR.parent
if str(AL_DIR) not in sys.path:
    sys.path.insert(0, str(AL_DIR))

import al_config as cfg

ROOT = cfg.ROOT
RUN_MD_DRIVER = STAGE_DIR / "run_md.py"


def require_file(path: Path) -> None:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size == 0:
        raise RuntimeError(f"Empty file: {path}")

def probe_complete() -> bool:
    if not cfg.PROBE_TRAJECTORY_FILE.is_file() or not cfg.PROBE_THERMO_FILE.is_file():
        return False
    try:
        frames = read(cfg.PROBE_TRAJECTORY_FILE, index=":")
    except Exception:
        return False
    return abs(len(frames) - cfg.PROBE_EXPECTED_MD_FRAMES) <= 2 and len(frames) > 0

def validate_inputs() -> None:
    require_file(cfg.PROBE_INPUT_STRUCTURE)
    require_file(cfg.CURRENT_GENERATION_SHORT_MODEL)
    require_file(RUN_MD_DRIVER)
    if cfg.PROBE_MD_PS <= 0.0:
        raise RuntimeError("PROBE_MD_PS must be positive.")
    if cfg.PROBE_TOTAL_MD_STEPS <= 0:
        raise RuntimeError("PROBE_TOTAL_MD_STEPS must be positive.")
    if cfg.PROBE_TOTAL_MD_STEPS % cfg.SAVE_EVERY != 0:
        raise RuntimeError("PROBE_TOTAL_MD_STEPS must be divisible by SAVE_EVERY.")

def main() -> None:
    cfg.validate_config()
    validate_inputs()
    print(f"FRESH CONVERGENCE PROBE v{cfg.ROUND}: system={cfg.SYSTEM} temp={getattr(cfg, 'PROBE_TEMPERATURE_K', cfg.TEMPERATURE_K)}K mode={cfg.PROBE_MODE}")
    if probe_complete():
        print("Fresh probe already complete.")
        return
    if cfg.PROBE_OUTPUT_DIR.exists():
        contents = list(cfg.PROBE_OUTPUT_DIR.iterdir())
        if contents:
            import shutil; shutil.rmtree(cfg.PROBE_OUTPUT_DIR)
    cfg.PROBE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "-u",
        str(RUN_MD_DRIVER),
        f"--mode={cfg.PROBE_MODE}",
        f"--structure={cfg.PROBE_INPUT_STRUCTURE}",
        f"--model={cfg.CURRENT_GENERATION_SHORT_MODEL}",
        f"--temperature={getattr(cfg, 'PROBE_TEMPERATURE_K', cfg.TEMPERATURE_K)}",
        f"--steps={cfg.PROBE_TOTAL_MD_STEPS}",
        f"--timestep-fs={cfg.TIMESTEP_FS}",
        f"--ttime-fs={cfg.TTIME_FS}",
        f"--save-every={cfg.SAVE_EVERY}",
        f"--seed={cfg.PROBE_MD_SEED}",
        f"--pair-capacity-factor={cfg.PAIR_CAPACITY_FACTOR}",
        f"--max-pairs={cfg.MAX_PAIRS}",
        f"--output={cfg.PROBE_OUTPUT_DIR}",
        f"--device={cfg.DEVICE}",
    ]
    if hasattr(cfg, "ACTIVE_METAL") and cfg.ACTIVE_METAL:
        command.append(f"--active-metal={cfg.ACTIVE_METAL}")
    if hasattr(cfg, "RCUT_METAL") and cfg.RCUT_METAL is not None:
        command.append(f"--rcut-metal={cfg.RCUT_METAL}")
    if hasattr(cfg, "ANION_CUTOFFS") and cfg.ANION_CUTOFFS:
        anion_str = ",".join(f"{k}:{v}" for k, v in cfg.ANION_CUTOFFS.items())
        command.append(f"--anion-cutoffs={anion_str}")

    if cfg.PROBE_MODE == "constp":
        command.extend([
            f"--bottom-external-chi={cfg.PROBE_BOTTOM_EXTERNAL_CHI}",
            f"--upper-external-chi={cfg.PROBE_UPPER_EXTERNAL_CHI}",
        ])
    elif cfg.PROBE_MODE != "constq":
        raise RuntimeError(f"Unsupported PROBE_MODE: {cfg.PROBE_MODE}")

    result = subprocess.run(command, cwd=AL_DIR)
    if result.returncode != 0:
        raise RuntimeError("Fresh convergence probe failed.")
    if not probe_complete():
        raise RuntimeError("Fresh probe finished but output is incomplete.")
    frames = read(cfg.PROBE_TRAJECTORY_FILE, index=":")
    print(f"FRESH PROBE COMPLETE: frames={len(frames)} traj={cfg.PROBE_TRAJECTORY_FILE}")

if __name__ == "__main__":
    main()
