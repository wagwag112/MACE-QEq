#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from ase.io import read, write

SCRIPT_DIR = Path(__file__).resolve().parent
AL_DIR = SCRIPT_DIR.parent
if str(AL_DIR) not in sys.path:
    sys.path.insert(0, str(AL_DIR))

from al_config import (
    ROUND,
    PARENT_ROUND,
    SYSTEM,
    TEMPERATURE_K,
    MD_PS,
    SELECTION_SCORE_FILE,
    SELECTION_TRAJECTORY_FILE,
    FRAMES_DIR,
    SELECTION_MANIFEST,
    N_HIGH,
    N_COVERAGE,
    N_HOLDOUT,
    N_ENRICH,
    N_STANDARD_ENRICH,
    N_SELECTED,
    HIGH_MIN_SPACING_PS,
    COVERAGE_MIN_SPACING_PS,
    MIN_HOLDOUT_DISTANCE_PS,
    COVERAGE_WINDOWS,
    HOLDOUT_WINDOWS,
)

SCORE_REQUIRED_COLUMNS = {
    "frame",
    "time_ps",
    "sigma_F_mean_eV_A",
    "sigma_F_p95_eV_A",
    "sigma_F_max_eV_A",
    "uncertainty_rank",
    "worst_atom",
    "worst_element",
}

CLEAN_ARRAY_KEYS = ["qeq_charges", "forces"]

def far_enough(time_ps, selected_times, min_spacing):
    return all(abs(time_ps - prev) >= min_spacing for prev in selected_times)

def validate_inputs(scores, frames):
    if len(scores) != len(frames):
        raise RuntimeError(f"Score count {len(scores)} != trajectory count {len(frames)}")
    missing = SCORE_REQUIRED_COLUMNS - set(scores.columns)
    if missing:
        raise RuntimeError(f"Missing score columns: {sorted(missing)}")
    if not np.array_equal(np.arange(len(scores), dtype=int), scores["frame"].to_numpy(dtype=int)):
        raise RuntimeError("Frame ordering mismatch between score file and trajectory.")
    if scores["frame"].duplicated().any() or not np.all(np.isfinite(scores["sigma_F_p95_eV_A"].to_numpy(dtype=float))):
        raise RuntimeError("Corrupted score file detected.")

def select_high_uncertainty(scores, excluded_frames):
    ranked = scores[~scores["frame"].isin(excluded_frames)].sort_values(["sigma_F_p95_eV_A", "frame"], ascending=[False, True])
    selected, selected_times = [], []
    for _, row in ranked.iterrows():
        frame = int(row["frame"])
        time_ps = float(row["time_ps"])
        if time_ps <= 0.0 or not far_enough(time_ps, selected_times, HIGH_MIN_SPACING_PS):
            continue
        selected.append(row.copy())
        selected_times.append(time_ps)
        excluded_frames.add(frame)
        if len(selected) == N_HIGH:
            break
    if len(selected) != N_HIGH:
        raise RuntimeError(f"Selected only {len(selected)}/{N_HIGH} high-uncertainty frames.")
    return selected

def best_available_in_window(scores, start_ps, end_ps, excluded_frames, selected_times):
    subset = scores[(scores["time_ps"] >= start_ps) & (scores["time_ps"] <= end_ps) & (~scores["frame"].isin(excluded_frames))].copy()
    subset = subset.sort_values(["sigma_F_p95_eV_A", "frame"], ascending=[False, True])
    for _, row in subset.iterrows():
        if far_enough(float(row["time_ps"]), selected_times, COVERAGE_MIN_SPACING_PS):
            return row.copy()
    raise RuntimeError(f"No suitable frame in coverage window {start_ps}-{end_ps} ps.")

def select_coverage(scores, excluded_frames, existing_rows):
    selected = []
    selected_times = [float(r["time_ps"]) for r in existing_rows]
    for start_ps, end_ps in COVERAGE_WINDOWS:
        row = best_available_in_window(scores, start_ps, end_ps, excluded_frames, selected_times)
        selected.append(row)
        excluded_frames.add(int(row["frame"]))
        selected_times.append(float(row["time_ps"]))
    if len(selected) != N_COVERAGE:
        raise RuntimeError(f"Selected only {len(selected)}/{N_COVERAGE} coverage frames.")
    return selected

def select_holdout(scores, excluded_frames, enrichment_rows):
    selected = []
    enrichment_times = [float(r["time_ps"]) for r in enrichment_rows]
    holdout_times = []
    for start_ps, end_ps in HOLDOUT_WINDOWS:
        subset = scores[(scores["time_ps"] >= start_ps) & (scores["time_ps"] <= end_ps) & (~scores["frame"].isin(excluded_frames))].copy()
        subset = subset.sort_values(["sigma_F_p95_eV_A", "frame"], ascending=[False, True])
        chosen = None
        for _, row in subset.iterrows():
            t = float(row["time_ps"])
            if far_enough(t, enrichment_times, MIN_HOLDOUT_DISTANCE_PS) and far_enough(t, holdout_times, MIN_HOLDOUT_DISTANCE_PS):
                chosen = row.copy()
                break
        if chosen is None:
            raise RuntimeError(f"No suitable holdout frame in {start_ps}-{end_ps} ps.")
        selected.append(chosen)
        excluded_frames.add(int(chosen["frame"]))
        holdout_times.append(float(chosen["time_ps"]))
    if len(selected) != N_HOLDOUT:
        raise RuntimeError(f"Selected only {len(selected)}/{N_HOLDOUT} holdout frames.")
    return selected

def clean_atoms(atoms):
    clean = atoms.copy()
    clean.calc = None
    for k in CLEAN_ARRAY_KEYS:
        if k in clean.arrays:
            del clean.arrays[k]
    return clean

def build_manifest(high_rows, coverage_rows, holdout_rows):
    rows = []
    for role, stype, g in [("train", "high_uncertainty", high_rows), ("train", "coverage", coverage_rows), ("holdout", "holdout", holdout_rows)]:
        for r in g:
            rows.append({
                "round": ROUND,
                "parent_round": PARENT_ROUND,
                "system": SYSTEM,
                "temperature_K": TEMPERATURE_K,
                "md_duration_ps": MD_PS,
                "role": role,
                "selection_type": stype,
                "frame": int(r["frame"]),
                "time_ps": float(r["time_ps"]),
                "sigma_F_mean_eV_A": float(r["sigma_F_mean_eV_A"]),
                "sigma_F_p95_eV_A": float(r["sigma_F_p95_eV_A"]),
                "sigma_F_max_eV_A": float(r["sigma_F_max_eV_A"]),
                "uncertainty_rank": int(r["uncertainty_rank"]),
                "worst_atom": int(r["worst_atom"]),
                "worst_element": str(r["worst_element"]),
            })
    manifest = pd.DataFrame(rows)
    if len(manifest) != N_SELECTED or manifest["frame"].duplicated().any():
        raise RuntimeError("Manifest construction invariant violated.")
    return manifest

def export_structures(manifest, frames):
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    for p in FRAMES_DIR.glob("*.vasp"):
        p.unlink()
    filenames = []
    for _, row in manifest.iterrows():
        frame = int(row["frame"])
        time_ps = float(row["time_ps"])
        sigma = float(row["sigma_F_p95_eV_A"])
        role = str(row["role"])
        atoms = clean_atoms(frames[frame])
        fn = f"{role}_{time_ps:05.1f}ps_f{frame:03d}_u{sigma:.4f}.vasp"
        write(str(FRAMES_DIR / fn), atoms, format="vasp", direct=True, vasp5=True, sort=False)
        filenames.append(fn)
    manifest = manifest.copy()
    manifest["filename"] = filenames
    return manifest

def main():
    if N_HIGH + N_COVERAGE != N_STANDARD_ENRICH or N_STANDARD_ENRICH != N_ENRICH or N_ENRICH + N_HOLDOUT != N_SELECTED:
        raise RuntimeError("Quota arithmetic invariant mismatch.")
    if not SELECTION_SCORE_FILE.exists() or not SELECTION_TRAJECTORY_FILE.exists():
        raise FileNotFoundError(f"Missing inputs: {SELECTION_SCORE_FILE} or {SELECTION_TRAJECTORY_FILE}")
    scores = pd.read_csv(SELECTION_SCORE_FILE)
    frames = read(SELECTION_TRAJECTORY_FILE, index=":")
    validate_inputs(scores, frames)
    excluded = set()
    high_rows = select_high_uncertainty(scores, excluded)
    coverage_rows = select_coverage(scores, excluded, high_rows)
    holdout_rows = select_holdout(scores, excluded, high_rows + coverage_rows)
    manifest = build_manifest(high_rows, coverage_rows, holdout_rows)
    train_frames = set(manifest.loc[manifest["role"] == "train", "frame"])
    holdout_frames = set(manifest.loc[manifest["role"] == "holdout", "frame"])
    if train_frames & holdout_frames:
        raise RuntimeError("Train/holdout data leakage detected.")
    manifest = export_structures(manifest, frames)
    manifest.sort_values(["role", "time_ps"]).reset_index(drop=True).to_csv(SELECTION_MANIFEST, index=False)
    print(f"CANDIDATE SELECTION COMPLETE: train={len(train_frames)} holdout={len(holdout_frames)} exported={FRAMES_DIR}")

if __name__ == "__main__":
    main()
