#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
from ase.io import read, write

SCRIPT_DIR = Path(__file__).resolve().parent
AL_DIR = SCRIPT_DIR.parent
if str(AL_DIR) not in sys.path:
    sys.path.insert(0, str(AL_DIR))

import al_config as cfg
from al_config import (
    DATA_DIR,
    ROUND,
    PARENT_ROUND,
    DFT_FILE,
    SHORT_FILE,
    PARENT_TRAIN_DFT,
    PARENT_TRAIN_SHORT,
    PARENT_VALID_DFT,
    PARENT_VALID_SHORT,
    ROUND_TRAIN_DFT,
    ROUND_TRAIN_SHORT,
    ROUND_HOLDOUT_DFT,
    ROUND_HOLDOUT_SHORT,
    OUT_TRAIN_DFT,
    OUT_TRAIN_SHORT,
    OUT_VALID_DFT,
    OUT_VALID_SHORT,
    N_SELECTED,
    N_ENRICH,
    N_HOLDOUT,
    EXPECTED_VALID,
)

POSITION_TOL = 1.0e-8
CELL_TOL = 1.0e-8

def load_frames(path: Path, label: str):
    if not path.is_file():
        raise FileNotFoundError(f"{label}: {path}")
    frames = read(path, index=":")
    if len(frames) == 0:
        raise RuntimeError(f"{label}: zero frames.")
    return frames

def make_source_map(frames, label: str):
    mapping = {}
    for i, a in enumerate(frames):
        src = a.info.get("source")
        if src is None:
            raise RuntimeError(f"{label} frame {i}: source metadata missing.")
        if src in mapping:
            raise RuntimeError(f"{label}: duplicate source {src}.")
        mapping[src] = a
    return mapping

def check_structure_match(dft_a, short_a, source: str, label: str):
    if len(dft_a) != len(short_a):
        raise RuntimeError(f"{label} {source}: atom count mismatch ({len(dft_a)} vs {len(short_a)}).")
    if dft_a.get_chemical_symbols() != short_a.get_chemical_symbols():
        raise RuntimeError(f"{label} {source}: species/order mismatch.")
    if not np.allclose(dft_a.positions, short_a.positions, rtol=0.0, atol=POSITION_TOL):
        raise RuntimeError(f"{label} {source}: atomic positions mismatch.")
    if not np.allclose(dft_a.cell.array, short_a.cell.array, rtol=0.0, atol=CELL_TOL):
        raise RuntimeError(f"{label} {source}: cell parameters mismatch.")
    if not np.array_equal(np.asarray(dft_a.pbc, dtype=bool), np.asarray(short_a.pbc, dtype=bool)):
        raise RuntimeError(f"{label} {source}: PBC mismatch.")

def check_ref_labels(frames, label: str):
    for i, a in enumerate(frames):
        src = a.info.get("source", f"frame_{i}")
        if "REF_energy" not in a.info or "REF_forces" not in a.arrays:
            raise RuntimeError(f"{label} {src}: REF_energy or REF_forces missing.")
        f = np.asarray(a.arrays["REF_forces"])
        if f.shape != (len(a), 3) or not np.isfinite(float(a.info["REF_energy"])) or not np.all(np.isfinite(f)):
            raise RuntimeError(f"{label} {src}: corrupted REF numerical values.")

def align_short_to_dft(dft_frames, short_frames, label: str):
    dft_map = make_source_map(dft_frames, f"{label} DFT")
    short_map = make_source_map(short_frames, f"{label} SHORT")
    if set(dft_map) != set(short_map):
        raise RuntimeError(f"{label}: DFT/Short sources mismatch.")
    aligned = []
    for dft_a in dft_frames:
        src = dft_a.info["source"]
        short_a = short_map[src]
        check_structure_match(dft_a, short_a, src, label)
        aligned.append(short_a)
    return aligned

def save_atomic_file(path: Path, frames):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    write(path, frames, format="extxyz")

def validate_dataset_readback(path: Path, expected_count: int, label: str):
    frames = read(path, index=":")
    if len(frames) != expected_count:
        raise RuntimeError(f"{label} ({path.name}): expected {expected_count} frames, found {len(frames)}.")
    sources = [a.info.get("source") for a in frames]
    if None in sources or len(sources) != len(set(sources)):
        raise RuntimeError(f"{label} ({path.name}): corrupted or duplicate source metadata.")
    check_ref_labels(frames, label)
    return frames

def main():
    if not DFT_FILE.is_file() or not SHORT_FILE.is_file():
        raise FileNotFoundError(f"Input round datasets missing: {DFT_FILE} or {SHORT_FILE}")

    curr_dft = load_frames(DFT_FILE, "CURRENT DFT")
    curr_short = load_frames(SHORT_FILE, "CURRENT SHORT")

    if ROUND == 0:
        target_v0 = getattr(cfg, "V0_TARGET_COUNT", 400)
        if len(curr_dft) != target_v0 or len(curr_short) != target_v0:
            raise RuntimeError(f"Expected {target_v0} frames in v0, found dft={len(curr_dft)}, short={len(curr_short)}.")
        curr_short = align_short_to_dft(curr_dft, curr_short, "AL-v0 IN-MEMORY ALIGNMENT")
        train_dft, train_short = [], []
        valid_dft, valid_short = [], []
        test_dft, test_short = [], []
        for d_a, s_a in zip(curr_dft, curr_short):
            role = d_a.info.get("al_role")
            if role == "train":
                train_dft.append(d_a); train_short.append(s_a)
            elif role == "valid":
                valid_dft.append(d_a); valid_short.append(s_a)
            elif role == "test":
                test_dft.append(d_a); test_short.append(s_a)
        save_atomic_file(OUT_TRAIN_DFT, train_dft)
        save_atomic_file(OUT_TRAIN_SHORT, train_short)
        save_atomic_file(OUT_VALID_DFT, valid_dft)
        save_atomic_file(OUT_VALID_SHORT, valid_short)
        save_atomic_file(DATA_DIR / "test_dft.extxyz", test_dft)
        save_atomic_file(DATA_DIR / "test_short.extxyz", test_short)
        validate_dataset_readback(OUT_TRAIN_DFT, len(train_dft), "v0 TRAIN DFT")
        validate_dataset_readback(OUT_TRAIN_SHORT, len(train_short), "v0 TRAIN SHORT")
        validate_dataset_readback(OUT_VALID_DFT, len(valid_dft), "v0 VALID DFT")
        validate_dataset_readback(OUT_VALID_SHORT, len(valid_short), "v0 VALID SHORT")
        validate_dataset_readback(DATA_DIR / "test_dft.extxyz", len(test_dft), "v0 TEST DFT")
        validate_dataset_readback(DATA_DIR / "test_short.extxyz", len(test_short), "v0 TEST SHORT")
        print(f"BOOTSTRAP v0 MERGE PASS: train={len(train_dft)} valid={len(valid_dft)} test={len(test_dft)}")
        return

    if len(curr_dft) != N_SELECTED or len(curr_short) != N_SELECTED:
        raise RuntimeError(f"Expected {N_SELECTED} frames in v{ROUND}, found dft={len(curr_dft)}, short={len(curr_short)}.")

    curr_short = align_short_to_dft(curr_dft, curr_short, f"AL-v{ROUND} IN-MEMORY ALIGNMENT")

    train_dft, train_short = [], []
    holdout_dft, holdout_short = [], []

    for dft_a, short_a in zip(curr_dft, curr_short):
        role = dft_a.info.get("al_role")
        if role != short_a.info.get("al_role"):
            raise RuntimeError(f"{dft_a.info.get('source')}: role mismatch between DFT and Short.")
        if role == "train":
            train_dft.append(dft_a)
            train_short.append(short_a)
        elif role == "holdout":
            holdout_dft.append(dft_a)
            holdout_short.append(short_a)
        else:
            raise RuntimeError(f"Unexpected al_role '{role}' in {dft_a.info.get('source')}")

    if len(train_dft) != N_ENRICH or len(holdout_dft) != N_HOLDOUT:
        raise RuntimeError(f"Split count mismatch: train={len(train_dft)}/{N_ENRICH}, holdout={len(holdout_dft)}/{N_HOLDOUT}")

    train_src = {a.info["source"] for a in train_dft}
    holdout_src = {a.info["source"] for a in holdout_dft}
    if train_src & holdout_src:
        raise RuntimeError("Data leakage: overlap between train and holdout partitions.")

    save_atomic_file(ROUND_TRAIN_DFT, train_dft)
    save_atomic_file(ROUND_TRAIN_SHORT, train_short)
    save_atomic_file(ROUND_HOLDOUT_DFT, holdout_dft)
    save_atomic_file(ROUND_HOLDOUT_SHORT, holdout_short)

    validate_dataset_readback(ROUND_TRAIN_DFT, N_ENRICH, f"v{ROUND} ROUND TRAIN DFT")
    validate_dataset_readback(ROUND_TRAIN_SHORT, N_ENRICH, f"v{ROUND} ROUND TRAIN SHORT")
    validate_dataset_readback(ROUND_HOLDOUT_DFT, N_HOLDOUT, f"v{ROUND} ROUND HOLDOUT DFT")
    validate_dataset_readback(ROUND_HOLDOUT_SHORT, N_HOLDOUT, f"v{ROUND} ROUND HOLDOUT SHORT")

    parent_dft = load_frames(PARENT_TRAIN_DFT, "PARENT TRAIN DFT")
    parent_short = load_frames(PARENT_TRAIN_SHORT, "PARENT TRAIN SHORT")
    valid_dft = load_frames(PARENT_VALID_DFT, "PARENT VALID DFT")
    valid_short = load_frames(PARENT_VALID_SHORT, "PARENT VALID SHORT")

    if len(valid_dft) != EXPECTED_VALID or len(valid_short) != EXPECTED_VALID:
        raise RuntimeError(f"Validation count mismatch: expected {EXPECTED_VALID}.")

    parent_short = align_short_to_dft(parent_dft, parent_short, "PARENT TRAIN ALIGNMENT")
    valid_short = align_short_to_dft(valid_dft, valid_short, "VALIDATION ALIGNMENT")

    p_src = {a.info["source"] for a in parent_dft}
    v_src = {a.info["source"] for a in valid_dft}

    if (p_src & train_src) or (p_src & v_src) or (train_src & v_src):
        raise RuntimeError("Data leakage detected between historical training, new enrichment, and validation sets.")

    final_dft = parent_dft + train_dft
    final_short = parent_short + train_short
    expected_total_train = len(parent_dft) + len(train_dft)

    final_short = align_short_to_dft(final_dft, final_short, f"FINAL v{ROUND} TRAIN ALIGNMENT")

    save_atomic_file(OUT_TRAIN_DFT, final_dft)
    save_atomic_file(OUT_TRAIN_SHORT, final_short)
    save_atomic_file(OUT_VALID_DFT, valid_dft)
    save_atomic_file(OUT_VALID_SHORT, valid_short)

    t_dft = validate_dataset_readback(OUT_TRAIN_DFT, expected_total_train, f"FINAL v{ROUND} TRAIN DFT")
    validate_dataset_readback(OUT_TRAIN_SHORT, expected_total_train, f"FINAL v{ROUND} TRAIN SHORT")
    v_dft = validate_dataset_readback(OUT_VALID_DFT, EXPECTED_VALID, f"FINAL v{ROUND} VALID DFT")
    validate_dataset_readback(OUT_VALID_SHORT, EXPECTED_VALID, f"FINAL v{ROUND} VALID SHORT")

    if {a.info["source"] for a in t_dft} & {a.info["source"] for a in v_dft}:
        raise RuntimeError("Final readback leakage: train and validation share sources.")

    print(f"SPLIT & MERGE PASS: v{ROUND} new_train={len(train_dft)} holdout={len(holdout_dft)} total_train={len(t_dft)} valid={len(v_dft)}")

if __name__ == "__main__":
    main()
