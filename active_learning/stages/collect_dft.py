#!/usr/bin/env python3
from __future__ import annotations
from collections import Counter
import csv
from pathlib import Path
import re
import sys
import numpy as np
from ase.io import read, write

SCRIPT_DIR = Path(__file__).resolve().parent
AL_DIR = SCRIPT_DIR.parent
if str(AL_DIR) not in sys.path:
    sys.path.insert(0, str(AL_DIR))

import al_config as cfg
from al_config import (
    ROUND,
    SYSTEM,
    DFT_CALC_DIR,
    DFT_FILE,
    DFT_REPORT,
    FRAMES_DIR,
    SELECTION_MANIFEST,
    N_SELECTED,
    N_ENRICH,
    N_HOLDOUT,
)

GEOMETRY_MATCH_TOL = 1.0e-3

MANIFEST_REQUIRED_FIELDS = {
    "filename",
    "role",
    "frame",
    "time_ps",
    "sigma_F_p95_eV_A",
    "system",
}

REPORT_FIELDNAMES = [
    "source",
    "role",
    "trajectory",
    "time_ps",
    "frame_index",
    "sigma_F_p95_eV_A",
    "system",
    "natoms",
    "energy_eV",
    "energy_per_atom_eV",
    "fmax_eV_A",
    "status",
    "path",
]

CLEAN_ARRAYS_KEYS = [
    "qeq_charges",
    "qeq_forces",
    "short_forces",
    "REF_forces",
    "dft_forces",
]

CLEAN_INFO_KEYS = [
    "REF_energy",
    "dft_energy",
    "short_energy",
    "qeq_energy",
]

READBACK_REQUIRED_INFO = ["REF_energy", "dft_energy"]
READBACK_REQUIRED_ARRAYS = ["REF_forces", "dft_forces"]

ROLE_ORDER_MAP = {
    "train": 0,
    "valid": 1,
    "test": 2,
    "diagnostic": 3,
    "": 3,
    "holdout": 4,
}

RE_STRUCTURE_SOURCE = re.compile(
    r"(train|holdout)_([0-9]+(?:\.[0-9]+)?)ps_f([0-9]+)_u([0-9]+(?:\.[0-9]+)?)"
)
RE_LEGACY_SOURCE = re.compile(r"(full|short)_([0-9]+(?:\.[0-9]+)?)ps")
RE_TIME_SOURCE = re.compile(r"([0-9]+(?:\.[0-9]+)?)ps")

def detect_system(atoms=None, xml_file=None):
    if xml_file is not None:
        cand = xml_file.parent.parent.name
        if cand in {cfg.ANODE, cfg.SE, cfg.SYSTEM}:
            return cand
    if atoms is not None and "system" in atoms.info:
        cand = str(atoms.info["system"]).strip()
        if cand in {cfg.ANODE, cfg.SE, cfg.SYSTEM}:
            return cand
    if atoms is not None and set(atoms.get_chemical_symbols()) == {cfg.ANODE}:
        return cfg.ANODE
    return cfg.SYSTEM
def parse_source(source):
    metadata = {
        "role": "",
        "trajectory": "",
        "time_ps": None,
        "frame_index": None,
        "sigma_F_p95": None,
    }
    match = RE_STRUCTURE_SOURCE.fullmatch(source)
    if match:
        metadata["role"] = match.group(1)
        metadata["trajectory"] = "exploration"
        metadata["time_ps"] = float(match.group(2))
        metadata["frame_index"] = int(match.group(3))
        metadata["sigma_F_p95"] = float(match.group(4))
        return metadata
    legacy = RE_LEGACY_SOURCE.fullmatch(source)
    if legacy:
        metadata["role"] = "diagnostic"
        metadata["trajectory"] = legacy.group(1)
        metadata["time_ps"] = float(legacy.group(2))
        return metadata
    time_match = RE_TIME_SOURCE.search(source)
    if time_match:
        metadata["time_ps"] = float(time_match.group(1))
    return metadata

def source_from_manifest_value(value):
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None
    name = Path(value).name
    if name.lower().endswith(".vasp"):
        name = Path(name).stem
    meta = parse_source(name)
    if meta["role"] in {"train", "holdout"}:
        return name
    return None

def load_manifest_metadata():
    if cfg.ROUND == 0:
        manifest_path = getattr(cfg, "V0_BOOTSTRAP_DIR", cfg.MD_ROOT / "v0_bootstrap") / "v0_selection_manifest.csv"
        expected_target = getattr(cfg, "V0_TARGET_COUNT", 400)
        allowed_roles = {"train", "valid", "test"}
    else:
        manifest_path = SELECTION_MANIFEST
        expected_target = N_SELECTED
        allowed_roles = {"train", "holdout"}

    if not manifest_path.is_file():
        raise FileNotFoundError(f"Selection manifest not found: {manifest_path}")

    metadata = {}
    with open(manifest_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row_index, row in enumerate(reader, start=2):
            filename = row["filename"].strip()
            source = Path(filename).stem
            if source in metadata:
                raise RuntimeError(f"Duplicate selected source in manifest: {source}")
            role = row["role"].strip()
            if role not in allowed_roles:
                raise RuntimeError(f"Invalid role in manifest row {row_index}: {role}")
            metadata[source] = {
                "role": role,
                "trajectory": "v0_bootstrap" if cfg.ROUND == 0 else (f"{cfg.EXPLORATION_MODE}_exploration" if hasattr(cfg, "EXPLORATION_MODE") else "exploration"),
                "time_ps": float(row["time_ps"]) if "time_ps" in row and row["time_ps"] else 0.0,
                "frame_index": int(row["frame"]) if "frame" in row and row["frame"] else None,
                "sigma_F_p95": float(row["sigma_F_p95_eV_A"]) if "sigma_F_p95_eV_A" in row and row["sigma_F_p95_eV_A"] else None,
                "system": row["system"].strip(),
            }
    if len(metadata) != expected_target:
        raise RuntimeError(
            f"Selection manifest contains {len(metadata)} structures; expected {expected_target}."
        )
    return metadata

def load_expected_sources():
    return list(load_manifest_metadata().keys())

def locate_selected_vaspruns(expected_sources):
    expected_set = set(expected_sources)
    matches = {source: [] for source in expected_sources}
    for xml_file in DFT_CALC_DIR.rglob("vasprun.xml"):
        source = xml_file.parent.name
        if source in expected_set:
            matches[source].append(xml_file)
    missing = [source for source, paths in matches.items() if len(paths) == 0]
    ambiguous = {source: paths for source, paths in matches.items() if len(paths) > 1}
    if missing:
        lines = "\n".join(f"  - {source}" for source in missing)
        raise RuntimeError(
            f"Selected DFT calculations are incomplete.\nMissing vasprun.xml for {len(missing)}/{len(expected_sources)} structures:\n{lines}"
        )
    if ambiguous:
        details = []
        for source, paths in ambiguous.items():
            details.append(f"  {source}:")
            details.extend(f"    {path}" for path in paths)
        raise RuntimeError(
            "Multiple vasprun.xml files match the same selected structure:\n" + "\n".join(details)
        )
    return [matches[source][0] for source in expected_sources]

def force_max(forces):
    forces = np.asarray(forces, dtype=np.float64)
    return float(np.max(np.linalg.norm(forces, axis=1)))

def clean_atoms(atoms):
    clean = atoms.copy()
    clean.calc = None
    for key in CLEAN_ARRAYS_KEYS:
        if key in clean.arrays:
            del clean.arrays[key]
    for key in CLEAN_INFO_KEYS:
        clean.info.pop(key, None)
    return clean

def verify_geometry_matches_export(atoms, source, tol=GEOMETRY_MATCH_TOL):
    if cfg.ROUND == 0:
        cal_root = cfg.ROOT / "DFT" / "v0" / "cal"
        poscars = list(cal_root.rglob(f"{source}/POSCAR"))
        if not poscars:
            raise RuntimeError(f"{source}: POSCAR not found under {cal_root}")
        export_path = poscars[0]
    else:
        export_path = FRAMES_DIR / f"{source}.vasp"
        if not export_path.is_file():
            raise RuntimeError(f"{source}: exported candidate structure not found at {export_path}")
    exported = read(export_path)
    if len(atoms) != len(exported):
        raise RuntimeError(f"{source}: vasprun atom count {len(atoms)} != export {len(exported)}")
    if atoms.get_chemical_symbols() != exported.get_chemical_symbols():
        raise RuntimeError(f"{source}: vasprun species/order mismatch with export")
    box = np.asarray(exported.cell.array, dtype=np.float64)
    box_inv = np.linalg.inv(box)
    dfrac = (atoms.positions - exported.positions) @ box_inv
    dfrac -= np.round(dfrac)
    dr = dfrac @ box
    max_disp = float(np.max(np.linalg.norm(dr, axis=1)))
    if max_disp > tol:
        raise RuntimeError(f"{source}: vasprun geometry differs from export by {max_disp:.4f} A (tol={tol} A)")

def read_vasprun(xml_file):
    atoms = read(xml_file, index=-1)
    natoms = len(atoms)
    if natoms <= 0:
        raise RuntimeError("Zero atoms.")
    energy = float(atoms.get_potential_energy())
    forces = np.asarray(atoms.get_forces(), dtype=np.float64)
    if not np.isfinite(energy):
        raise RuntimeError("Non-finite energy.")
    if forces.shape != (natoms, 3):
        raise RuntimeError(f"Invalid force shape: {forces.shape}")
    if not np.all(np.isfinite(forces)):
        raise RuntimeError("Non-finite forces.")
    return atoms, energy, forces

def sort_key(atoms):
    role = atoms.info.get("al_role", "")
    time_ps = atoms.info.get("time_ps", np.inf)
    try:
        time_ps = float(time_ps)
    except Exception:
        time_ps = np.inf
    return (ROLE_ORDER_MAP.get(role, 1), time_ps, str(atoms.info.get("source", "")))

def validate_role_counts(frames):
    roles = Counter(atoms.info.get("al_role", "") for atoms in frames)
    if cfg.ROUND == 0:
        exp_train = sum(spec["counts"]["train"] for spec in cfg.V0_SPECS.values())
        exp_valid = sum(spec["counts"]["valid"] for spec in cfg.V0_SPECS.values())
        exp_test = sum(spec["counts"]["test"] for spec in cfg.V0_SPECS.values())
        train_count = roles.get("train", 0)
        valid_count = roles.get("valid", 0)
        test_count = roles.get("test", 0)
        if train_count != exp_train:
            raise RuntimeError(f"Expected {exp_train} train frames, found {train_count}.")
        if valid_count != exp_valid:
            raise RuntimeError(f"Expected {exp_valid} valid frames, found {valid_count}.")
        if test_count != exp_test:
            raise RuntimeError(f"Expected {exp_test} test frames, found {test_count}.")
        if train_count + valid_count + test_count != len(frames):
            raise RuntimeError("Unexpected al_role values in v0.")
    else:
        train_count = roles.get("train", 0)
        holdout_count = roles.get("holdout", 0)
        if train_count != N_ENRICH:
            raise RuntimeError(f"Expected {N_ENRICH} train frames, found {train_count}.")
        if holdout_count != N_HOLDOUT:
            raise RuntimeError(f"Expected {N_HOLDOUT} holdout frames, found {holdout_count}.")
        if train_count + holdout_count != len(frames):
            raise RuntimeError("Unexpected al_role values found.")

def validate_readback(frames):
    for i, atoms in enumerate(frames):
        source = atoms.info.get("source", "UNKNOWN")
        for key in READBACK_REQUIRED_INFO:
            if key not in atoms.info:
                raise RuntimeError(f"Frame {i} {source}: {key} missing.")
        for key in READBACK_REQUIRED_ARRAYS:
            if key not in atoms.arrays:
                raise RuntimeError(f"Frame {i} {source}: {key} missing.")
        if atoms.arrays["REF_forces"].shape != (len(atoms), 3):
            raise RuntimeError(f"{source}: invalid REF_forces shape.")

def main():
    DFT_FILE.parent.mkdir(parents=True, exist_ok=True)
    DFT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    if not DFT_CALC_DIR.exists():
        raise FileNotFoundError(DFT_CALC_DIR)
    manifest_metadata = load_manifest_metadata()
    expected_target = getattr(cfg, "V0_TARGET_COUNT", 400) if cfg.ROUND == 0 else N_SELECTED
    expected_sources = list(manifest_metadata.keys())
    xml_files = locate_selected_vaspruns(expected_sources)

    print(f"COLLECT AL-v{ROUND} VASP DFT: input={DFT_CALC_DIR} xml={len(xml_files)} expected={expected_target}")
    frames = []
    report = []
    seen_sources = set()

    for i, xml_file in enumerate(xml_files, start=1):
        source = xml_file.parent.name
        meta = manifest_metadata[source]
        try:
            if source in seen_sources:
                raise RuntimeError(f"Duplicate source: {source}")
            if meta["role"] not in {"train", "holdout", "valid", "test"}:
                raise RuntimeError(f"Invalid or missing role in source name: {source}")
            atoms, energy, forces = read_vasprun(xml_file)
            verify_geometry_matches_export(atoms, source)
            for binary_name in ["WAVECAR", "CHGCAR", "CHG"]:
                bf = xml_file.parent / binary_name
                if bf.is_file():
                    try:
                        bf.unlink()
                    except Exception:
                        pass
            folder_system = xml_file.parent.parent.name
            folder_system = cfg.SE if folder_system == "LPSC" else (cfg.SYSTEM if folder_system == "Li_LPSC" else folder_system)
            meta_system = meta.get("system", "").strip()
            meta_system = cfg.SE if meta_system == "LPSC" else (cfg.SYSTEM if meta_system == "Li_LPSC" else meta_system)

            if folder_system in {cfg.ANODE, cfg.SE, cfg.SYSTEM}:
                system = folder_system
            elif meta_system in {cfg.ANODE, cfg.SE, cfg.SYSTEM}:
                system = meta_system
            else:
                system = detect_system(atoms, xml_file)
            if not system or system == "unknown":
                raise RuntimeError(f"Unknown system for calculation: {source}")
            clean = clean_atoms(atoms)
            clean.info["source"] = source
            clean.info["system"] = system
            clean.info["al_round"] = int(ROUND)
            clean.info["al_role"] = meta["role"]
            if meta["trajectory"]:
                clean.info["trajectory"] = meta["trajectory"]
            if meta["time_ps"] is not None:
                clean.info["time_ps"] = float(meta["time_ps"])
            if meta["frame_index"] is not None:
                clean.info["md_frame"] = int(meta["frame_index"])
            if meta["sigma_F_p95"] is not None:
                clean.info["sigma_F_p95"] = float(meta["sigma_F_p95"])
            clean.info["dft_energy"] = energy
            clean.info["REF_energy"] = energy
            clean.arrays["dft_forces"] = forces.copy()
            clean.arrays["REF_forces"] = forces.copy()
            frames.append(clean)
            seen_sources.add(source)
            fmax = force_max(forces)
            report.append({
                "source": source,
                "role": meta["role"],
                "trajectory": meta["trajectory"],
                "time_ps": meta["time_ps"] if meta["time_ps"] is not None else "",
                "frame_index": meta["frame_index"] if meta["frame_index"] is not None else "",
                "sigma_F_p95_eV_A": meta["sigma_F_p95"] if meta["sigma_F_p95"] is not None else "",
                "system": system,
                "natoms": len(atoms),
                "energy_eV": energy,
                "energy_per_atom_eV": energy / len(atoms),
                "fmax_eV_A": fmax,
                "status": "PASS",
                "path": str(xml_file),
            })
            print(f"[{i:2d}/{len(xml_files):2d}] PASS {source:34s} E/N={energy/len(atoms):11.6f} Fmax={fmax:8.4f}")
        except Exception as exc:
            report.append({
                "source": source,
                "role": meta.get("role", ""),
                "trajectory": meta.get("trajectory", ""),
                "time_ps": meta.get("time_ps", "") if meta.get("time_ps") is not None else "",
                "frame_index": meta.get("frame_index", "") if meta.get("frame_index") is not None else "",
                "sigma_F_p95_eV_A": meta.get("sigma_F_p95", "") if meta.get("sigma_F_p95") is not None else "",
                "system": "",
                "natoms": "",
                "energy_eV": "",
                "energy_per_atom_eV": "",
                "fmax_eV_A": "",
                "status": f"FAIL: {exc}",
                "path": str(xml_file),
            })
            print(f"[{i:2d}/{len(xml_files):2d}] FAIL {source}: {exc}")

    with open(DFT_REPORT, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPORT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(report)

    failed_count = len(xml_files) - len(frames)
    expected_target = getattr(cfg, "V0_TARGET_COUNT", 400) if cfg.ROUND == 0 else N_SELECTED
    if failed_count != 0:
        raise RuntimeError(f"{failed_count} DFT calculations failed. See report: {DFT_REPORT}")
    if len(frames) != expected_target:
        raise RuntimeError(f"Expected {expected_target} valid frames, collected {len(frames)}.")
    sources = [atoms.info["source"] for atoms in frames]
    if len(sources) != len(set(sources)):
        raise RuntimeError("Duplicate source names.")

    validate_role_counts(frames)
    frames = sorted(frames, key=sort_key)
    if DFT_FILE.exists():
        DFT_FILE.unlink()
    write(DFT_FILE, frames, format="extxyz")

    check = read(DFT_FILE, index=":")
    if len(check) != len(frames):
        raise RuntimeError("Read-back frame count mismatch.")
    validate_readback(check)
    validate_role_counts(check)
    print(f"COLLECT COMPLETE: {len(check)} frames written to {DFT_FILE}")

if __name__ == "__main__":
    main()
