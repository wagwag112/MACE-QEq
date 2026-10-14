#!/usr/bin/env python3
from __future__ import annotations
import argparse
import gc
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ase.io import read
from mace.calculators import MACECalculator

SCRIPT_DIR = Path(__file__).resolve().parent
AL_DIR = SCRIPT_DIR.parent
if str(AL_DIR) not in sys.path:
    sys.path.insert(0, str(AL_DIR))

from al_config import (
    SELECTION_TRAJECTORY_FILE as TRAJECTORY_FILE,
    TRAINED_MODELS_ROOT,
    ROOT,
    ROUND,
    PARENT_ROUND,
    SYSTEM,
    TEMPERATURE_K,
    MD_PS,
    DEVICE,
    SCORING_DIR,
    SCORING_FIGURE_DIR,
    SCORE_FILE,
    TOP_UNCERTAINTY_FILE,
    FRAME_INTERVAL_PS,
    EXPECTED_MD_FRAMES,
    BASE_SEED,
    BASE_COMMITTEE_MODEL,
    COMMITTEE_ROOT,
    COMMITTEE_SEEDS,
)

TOP_UNCERTAINTY_COUNT = 50
PROGRESS_LOG_INTERVAL = 25
PERCENTILE_P95 = 95.0
PERCENTILE_P99 = 99.0

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=Path, default=None)
    parser.add_argument("--expected-frames", type=int, default=None)
    parser.add_argument("--frame-interval-ps", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--committee-version", type=int, default=None)
    return parser.parse_args()

def apply_cli_overrides(args):
    global TRAJECTORY_FILE, EXPECTED_MD_FRAMES, FRAME_INTERVAL_PS
    global SCORING_DIR, SCORING_FIGURE_DIR, SCORE_FILE, TOP_UNCERTAINTY_FILE
    global BASE_COMMITTEE_MODEL, COMMITTEE_ROOT
    if args.trajectory is not None:
        TRAJECTORY_FILE = args.trajectory.resolve()
    if args.expected_frames is not None:
        EXPECTED_MD_FRAMES = args.expected_frames
    if args.frame_interval_ps is not None:
        FRAME_INTERVAL_PS = args.frame_interval_ps
    if args.output_dir is not None:
        SCORING_DIR = args.output_dir.resolve()
        SCORING_FIGURE_DIR = SCORING_DIR / "figures"
        SCORE_FILE = SCORING_DIR / "committee_scores.csv"
        TOP_UNCERTAINTY_FILE = SCORING_DIR / "top50_uncertainty.csv"
    if args.committee_version is not None:
        v = args.committee_version
        BASE_COMMITTEE_MODEL = TRAINED_MODELS_ROOT / f"train_short_v{v}" / "checkpoints" / f"MACE_short_v{v}_run-{BASE_SEED}.model"
        COMMITTEE_ROOT = TRAINED_MODELS_ROOT / f"committee_short_v{v}"

def find_seed_model(seed):
    if seed == BASE_SEED:
        if not BASE_COMMITTEE_MODEL.exists():
            raise FileNotFoundError(f"Base model missing: {BASE_COMMITTEE_MODEL}")
        return BASE_COMMITTEE_MODEL
    checkpoint_dir = COMMITTEE_ROOT / f"seed_{seed}" / "checkpoints"
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Committee dir missing: {checkpoint_dir}")
    candidates = sorted(p for p in checkpoint_dir.glob("*.model") if "stagetwo" not in p.name.lower())
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one model for seed {seed}, found {len(candidates)}")
    return candidates[0]

def build_committee():
    if len(COMMITTEE_SEEDS) < 2 or len(COMMITTEE_SEEDS) != len(set(COMMITTEE_SEEDS)):
        raise RuntimeError("Invalid committee seed configuration.")
    if BASE_SEED not in COMMITTEE_SEEDS:
        raise RuntimeError("BASE_SEED is missing in COMMITTEE_SEEDS.")
    models = [(s, find_seed_model(s)) for s in COMMITTEE_SEEDS]
    paths = [str(p.resolve()) for _, p in models]
    if len(paths) != len(set(paths)):
        raise RuntimeError("Duplicate committee model paths.")
    return models

def load_trajectory():
    global EXPECTED_MD_FRAMES
    if not TRAJECTORY_FILE.exists():
        raise FileNotFoundError(f"Trajectory missing: {TRAJECTORY_FILE}")
    frames = read(TRAJECTORY_FILE, index=":")
    if abs(len(frames) - EXPECTED_MD_FRAMES) > 2 or len(frames) == 0:
        raise RuntimeError(f"Expected {EXPECTED_MD_FRAMES} frames, found {len(frames)}.")
    EXPECTED_MD_FRAMES = len(frames)
    natoms = len(frames[0])
    symbols = frames[0].get_chemical_symbols()
    for i, a in enumerate(frames):
        if len(a) != natoms or a.get_chemical_symbols() != symbols:
            raise RuntimeError(f"Trajectory invariant failure at frame {i}")
    return frames

def predict_member(frames, seed, model_path, member_id):
    calc = MACECalculator(model_paths=[str(model_path)], device=DEVICE)
    predictions = []
    for i, a in enumerate(frames, start=1):
        work = a.copy()
        work.calc = calc
        forces = np.asarray(work.get_forces(), dtype=np.float64)
        if forces.shape != (len(a), 3) or not np.all(np.isfinite(forces)):
            raise RuntimeError(f"Member {member_id} frame {i-1} force calculation error.")
        predictions.append(forces)
        if i % PROGRESS_LOG_INTERVAL == 0 or i == len(frames):
            print(f"Member {member_id} (seed {seed}) predicted {i:4d}/{len(frames)}")
    del calc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return predictions

def get_time_ps(atoms, frame_index):
    if "time_ps" in atoms.info:
        try:
            return float(atoms.info["time_ps"])
        except Exception:
            pass
    return frame_index * FRAME_INTERVAL_PS

def score_frames(frames, predictions):
    rows = []
    n_members = len(predictions)
    for frame_index, atoms in enumerate(frames):
        member_forces = np.stack([predictions[m][frame_index] for m in range(n_members)], axis=0)
        mean_force = np.mean(member_forces, axis=0)
        deviation = member_forces - mean_force[None, :, :]
        sigma_atom = np.sqrt(np.mean(np.sum(deviation ** 2, axis=2), axis=0))
        if not np.all(np.isfinite(sigma_atom)):
            raise RuntimeError(f"Non-finite uncertainty at frame {frame_index}")
        worst_atom = int(np.argmax(sigma_atom))
        rows.append({
            "frame": frame_index,
            "time_ps": get_time_ps(atoms, frame_index),
            "natoms": len(atoms),
            "sigma_F_mean_eV_A": float(np.mean(sigma_atom)),
            "sigma_F_median_eV_A": float(np.median(sigma_atom)),
            "sigma_F_p95_eV_A": float(np.percentile(sigma_atom, PERCENTILE_P95)),
            "sigma_F_p99_eV_A": float(np.percentile(sigma_atom, PERCENTILE_P99)),
            "sigma_F_max_eV_A": float(np.max(sigma_atom)),
            "worst_atom": worst_atom,
            "worst_element": atoms.get_chemical_symbols()[worst_atom],
            "mean_force_magnitude_eV_A": float(np.mean(np.linalg.norm(mean_force, axis=1))),
        })
    df = pd.DataFrame(rows)
    df["uncertainty_rank"] = df["sigma_F_p95_eV_A"].rank(ascending=False, method="min").astype(int)
    return df.sort_values("frame").reset_index(drop=True)

def make_figures(df):
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.plot(df["time_ps"], df["sigma_F_p95_eV_A"], linewidth=1.1)
    ax.set_xlabel("MD time (ps)")
    ax.set_ylabel("Committee sigma F p95 (eV/A)")
    ax.set_title(f"AL-v{ROUND} {SYSTEM} {TEMPERATURE_K}K committee uncertainty")
    fig.tight_layout()
    fig.savefig(SCORING_FIGURE_DIR / "sigma_p95_vs_time.png", dpi=300)
    plt.close(fig)

def main():
    args = parse_args()
    apply_cli_overrides(args)
    SCORING_DIR.mkdir(parents=True, exist_ok=True)
    SCORING_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    committee = build_committee()
    frames = load_trajectory()
    predictions = [predict_member(frames, s, p, i) for i, (s, p) in enumerate(committee)]
    df = score_frames(frames, predictions)
    df.to_csv(SCORE_FILE, index=False)
    top = df.sort_values("sigma_F_p95_eV_A", ascending=False).head(min(TOP_UNCERTAINTY_COUNT, len(df))).copy()
    top.to_csv(TOP_UNCERTAINTY_FILE, index=False)
    make_figures(df)
    print(f"COMMITTEE SCORING COMPLETE: frames={len(df)} score_file={SCORE_FILE}")

if __name__ == "__main__":
    main()
