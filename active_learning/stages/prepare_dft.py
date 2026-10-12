#!/usr/bin/env python3
from __future__ import annotations
import csv
from pathlib import Path
import shutil
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
AL_DIR = SCRIPT_DIR.parent
if str(AL_DIR) not in sys.path:
    sys.path.insert(0, str(AL_DIR))

import al_config as cfg

MANIFEST_REQUIRED_COLUMNS = {
    "round",
    "parent_round",
    "system",
    "role",
    "selection_type",
    "frame",
    "time_ps",
    "sigma_F_p95_eV_A",
    "filename",
}

def require_file(path: Path) -> None:
    path = Path(path)
    if not path.is_file():
        raise RuntimeError(f"Required file does not exist: {path}")
    if path.stat().st_size == 0:
        raise RuntimeError(f"Required file is empty: {path}")

def load_manifest() -> list[dict[str, str]]:
    manifest = Path(cfg.SELECTION_MANIFEST)
    require_file(manifest)
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        rows = list(reader)
    missing = MANIFEST_REQUIRED_COLUMNS - fieldnames
    if missing:
        raise RuntimeError(f"Selection manifest missing columns: {sorted(missing)}")
    if len(rows) != cfg.N_SELECTED:
        raise RuntimeError(f"Manifest count {len(rows)} != expected {cfg.N_SELECTED}")
    return rows

def validate_manifest(rows: list[dict[str, str]]) -> None:
    filenames = []
    train_count = 0
    holdout_count = 0
    for i, row in enumerate(rows):
        if int(row["round"]) != cfg.ROUND:
            raise RuntimeError(f"Manifest row {i}: round mismatch.")
        if int(row["parent_round"]) != cfg.PARENT_ROUND:
            raise RuntimeError(f"Manifest row {i}: parent-round mismatch.")
        if row["system"].strip() != cfg.SYSTEM:
            raise RuntimeError(f"Manifest row {i}: system mismatch.")
        filename = row["filename"].strip()
        if not filename or not filename.endswith(".vasp"):
            raise RuntimeError(f"Manifest row {i}: invalid filename {filename}")
        filenames.append(filename)
        role = row["role"].strip()
        if role == "train":
            train_count += 1
        elif role == "holdout":
            holdout_count += 1
        else:
            raise RuntimeError(f"Manifest row {i}: unknown role {role}")
    if len(filenames) != len(set(filenames)):
        raise RuntimeError("Duplicate filenames in manifest.")
    if train_count != cfg.N_ENRICH:
        raise RuntimeError(f"Expected {cfg.N_ENRICH} train, found {train_count}")
    if holdout_count != cfg.N_HOLDOUT:
        raise RuntimeError(f"Expected {cfg.N_HOLDOUT} holdout, found {holdout_count}")

def validate_sources(rows: list[dict[str, str]]) -> None:
    frames_dir = Path(cfg.FRAMES_DIR)
    if not frames_dir.is_dir():
        raise RuntimeError(f"Frames directory missing: {frames_dir}")
    for row in rows:
        require_file(frames_dir / row["filename"])

def output_complete(rows: list[dict[str, str]]) -> bool:
    cal_root = getattr(cfg, "DFT_CALC_DIR", cfg.ROOT / "DFT" / f"v{cfg.ROUND}" / "cal")
    if not cal_root.is_dir():
        return False
    poscars = list(cal_root.rglob("POSCAR"))
    return len(poscars) == len(rows)

def prepare(rows: list[dict[str, str]]) -> list[Path]:
    cal_root = getattr(cfg, "DFT_CALC_DIR", cfg.ROOT / "DFT" / f"v{cfg.ROUND}" / "cal")
    if output_complete(rows):
        return sorted(cal_root.rglob("POSCAR"))
    output_dir = Path(cfg.DFT_SELECTED_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    cal_root = getattr(cfg, "DFT_CALC_DIR", cfg.ROOT / "DFT" / f"v{cfg.ROUND}" / "cal")
    cal_root.mkdir(parents=True, exist_ok=True)
    outputs = []
    for row in rows:
        fn = row["filename"]
        source_name = Path(fn).stem
        system = row["system"].strip()
        source_file = Path(cfg.FRAMES_DIR) / fn
        shutil.copy2(source_file, output_dir / fn)
        job_dir = cal_root / system / source_name
        job_dir.mkdir(parents=True, exist_ok=True)
        poscar_target = job_dir / "POSCAR"
        shutil.copy2(source_file, poscar_target)
        require_file(poscar_target)
        outputs.append(poscar_target)
    if not output_complete(rows):
        raise RuntimeError("Prepared DFT hand-off does not match manifest.")
    return outputs

def main() -> None:
    cfg.validate_config()
    rows = load_manifest()
    validate_manifest(rows)
    validate_sources(rows)
    outputs = prepare(rows)
    print(f"DFT HAND-OFF READY: AL v{cfg.ROUND} structures={len(outputs)} train={cfg.N_ENRICH} holdout={cfg.N_HOLDOUT}")

if __name__ == "__main__":
    main()
