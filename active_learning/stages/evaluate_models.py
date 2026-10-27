#!/usr/bin/env python3
from __future__ import annotations
import gc
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
from ase.io import read
from mace.calculators import MACECalculator

SCRIPT_DIR = Path(__file__).resolve().parent
AL_DIR = SCRIPT_DIR.parent
if str(AL_DIR) not in sys.path:
    sys.path.insert(0, str(AL_DIR))

import al_config as cfg

ROOT = cfg.ROOT
DATA_DIR = cfg.DATA_DIR
MD_ROOT = cfg.MD_ROOT
ROUND = cfg.ROUND
PARENT_ROUND = cfg.PARENT_ROUND
DEVICE = cfg.DEVICE

START_HOLDOUT_GEN = 1
POSITION_TOL = 1.0e-8
CELL_TOL = 1.0e-8
DECOMP_E_TOL = 1.0e-6
DECOMP_F_TOL = 1.0e-10

OUTPUT_DIR = getattr(cfg, "EVALUATION_DIR", MD_ROOT / f"active_learning_v{ROUND}" / "validation" / "evaluation")
PREFIX = (
    f"v{ROUND if 'ROUND' in globals() else cfg.ROUND}_standalone"
    if (PARENT_ROUND if 'PARENT_ROUND' in globals() else cfg.PARENT_ROUND) < 0
    else f"v{PARENT_ROUND if 'PARENT_ROUND' in globals() else cfg.PARENT_ROUND}_vs_v{ROUND if 'ROUND' in globals() else cfg.ROUND}"
)
SUMMARY_FILE = OUTPUT_DIR / f"{PREFIX}_summary.csv"
FRAME_FILE = OUTPUT_DIR / f"{PREFIX}_frames.csv"
ELEMENT_FILE = OUTPUT_DIR / f"{PREFIX}_elements.csv"
WORST_FILE = OUTPUT_DIR / f"{PREFIX}_worst_force_cases.csv"
CHANGE_FILE = OUTPUT_DIR / f"{PREFIX}_changes.csv"
TEST_DFT = DATA_DIR / "test_dft.extxyz"
TEST_SHORT = DATA_DIR / "test_short.extxyz"

CHANGE_METRICS = [
    "E_RMSE_full_meV_atom",
    "E_RMSE_shortQEq_meV_atom",
    "F_RMSE_full_eV_A",
    "F_RMSE_shortQEq_eV_A",
    "F_maxerr_full_eV_A",
    "F_maxerr_shortQEq_eV_A",
]

def rmse(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return float(np.sqrt(np.mean((x - y) ** 2)))

def mae(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return float(np.mean(np.abs(x - y)))

def flatten_forces(force_list):
    return np.concatenate([np.asarray(f, dtype=np.float64).reshape(-1) for f in force_list])

def get_system(atoms):
    metadata = atoms.info.get("system")
    if metadata is not None:
        metadata = str(metadata).strip()
        if metadata == "LPSC":
            metadata = getattr(cfg, "SE", "Li6PS5Cl") if cfg is not None else "Li6PS5Cl"
        elif metadata == "Li_LPSC":
            metadata = getattr(cfg, "SYSTEM", "Li_Li6PS5Cl") if cfg is not None else "Li_Li6PS5Cl"
        if cfg is not None and metadata in {cfg.ANODE, cfg.SE, cfg.SYSTEM}:
            return metadata
        if metadata and metadata.lower() != "unknown":
            return metadata

    symbols = set(atoms.get_chemical_symbols())
    anode = getattr(cfg, "ANODE", "Li") if cfg is not None else "Li"
    if symbols == {anode}:
        return anode

    cutoffs = getattr(cfg, "SE_ELEMENT_CUTOFFS", getattr(cfg, "ANION_CUTOFFS", {})) if cfg is not None else {}
    other_elements = symbols - {anode}
    if cutoffs and other_elements.issubset(cutoffs.keys()):
        try:
            from ase.neighborlist import neighbor_list
            from scipy.sparse import csr_matrix
            from scipy.sparse.csgraph import connected_components
            sym_arr = np.asarray(atoms.get_chemical_symbols())
            natoms = len(atoms)
            r_metal = float(getattr(cfg, "RCUT_METAL", 3.50))
            min_cluster = int(getattr(cfg, "MIN_CLUSTER_SIZE", 8))
            max_rc = max(r_metal, max(cutoffs[el] for el in other_elements))
            pair_i, pair_j, d = neighbor_list("ijd", atoms, max_rc, self_interaction=False)
            is_metal = (sym_arr == anode)
            other_bonds = np.zeros(len(pair_i), dtype=bool)
            for el in other_elements:
                other_bonds |= (is_metal[pair_i] & (sym_arr[pair_j] == el) & (d <= cutoffs[el]))
            cn_other = np.bincount(pair_i[other_bonds], minlength=natoms)
            is_cand = is_metal & (cn_other <= 1)
            if np.sum(is_cand) >= min_cluster:
                metal_bonds = is_cand[pair_i] & is_cand[pair_j] & (d <= r_metal)
                adj = csr_matrix((np.ones(np.sum(metal_bonds), dtype=bool), (pair_i[metal_bonds], pair_j[metal_bonds])), shape=(natoms, natoms))
                _, labels = connected_components(adj, directed=False)
                counts = np.bincount(labels[is_cand])
                if np.any(counts >= min_cluster):
                    return str(getattr(cfg, "SYSTEM", f"{anode}_{getattr(cfg, 'SE', 'SE')}"))
            return str(getattr(cfg, "SE", "SE"))
        except Exception:
            pass

    if cfg is not None and getattr(cfg, "SYSTEM", None):
        return str(cfg.SYSTEM)

    return "_".join(sorted(symbols))

def get_dft_energy(atoms):
    if "REF_energy" in atoms.info:
        return float(atoms.info["REF_energy"])
    if "dft_energy" in atoms.info:
        return float(atoms.info["dft_energy"])
    if atoms.calc is not None:
        return float(atoms.get_potential_energy())
    raise RuntimeError(f"DFT energy missing for {atoms.info.get('source', 'UNKNOWN')}")

def get_dft_forces(atoms):
    if "REF_forces" in atoms.arrays:
        return np.asarray(atoms.arrays["REF_forces"], dtype=np.float64)
    if "dft_forces" in atoms.arrays:
        return np.asarray(atoms.arrays["dft_forces"], dtype=np.float64)
    if atoms.calc is not None:
        return np.asarray(atoms.get_forces(), dtype=np.float64)
    raise RuntimeError(f"DFT forces missing for {atoms.info.get('source', 'UNKNOWN')}")

def get_short_energy(atoms):
    if "short_energy" in atoms.info:
        return float(atoms.info["short_energy"])
    raise RuntimeError(f"Short energy missing for {atoms.info.get('source', 'UNKNOWN')}")

def get_short_forces(atoms):
    if "short_forces" in atoms.arrays:
        return np.asarray(atoms.arrays["short_forces"], dtype=np.float64)
    raise RuntimeError(f"Short forces missing for {atoms.info.get('source', 'UNKNOWN')}")

def get_qeq_energy(atoms):
    if "qeq_energy" not in atoms.info:
        raise RuntimeError(f"QEq energy missing for {atoms.info.get('source', 'UNKNOWN')}")
    return float(atoms.info["qeq_energy"])

def get_qeq_forces(atoms):
    if "qeq_forces" not in atoms.arrays:
        raise RuntimeError(f"QEq forces missing for {atoms.info.get('source', 'UNKNOWN')}")
    return np.asarray(atoms.arrays["qeq_forces"], dtype=np.float64)

def make_map(frames, label):
    mapping = {}
    for i, atoms in enumerate(frames):
        source = atoms.info.get("source")
        if source is None:
            raise RuntimeError(f"{label} frame {i}: source missing.")
        if source in mapping:
            raise RuntimeError(f"{label}: duplicate source {source}")
        mapping[source] = atoms
    return mapping

def align_datasets(dft_frames, short_frames, label):
    dft_map = make_map(dft_frames, f"{label} DFT")
    short_map = make_map(short_frames, f"{label} SHORT")
    if set(dft_map) != set(short_map):
        raise RuntimeError(
            f"{label}: DFT/short source mismatch.\n"
            f"Missing short: {sorted(set(dft_map) - set(short_map))}\n"
            f"Missing DFT: {sorted(set(short_map) - set(dft_map))}"
        )
    dft_aligned, short_aligned = [], []
    for dft_atoms in dft_frames:
        source = dft_atoms.info["source"]
        short_atoms = short_map[source]
        if len(dft_atoms) != len(short_atoms):
            raise RuntimeError(f"{label}/{source}: atom-count mismatch.")
        if dft_atoms.get_chemical_symbols() != short_atoms.get_chemical_symbols():
            raise RuntimeError(f"{label}/{source}: atom-order mismatch.")
        if not np.allclose(dft_atoms.positions, short_atoms.positions, rtol=0.0, atol=POSITION_TOL):
            raise RuntimeError(f"{label}/{source}: position mismatch.")
        if not np.allclose(dft_atoms.cell.array, short_atoms.cell.array, rtol=0.0, atol=CELL_TOL):
            raise RuntimeError(f"{label}/{source}: cell mismatch.")
        if not np.array_equal(np.asarray(dft_atoms.pbc, dtype=bool), np.asarray(short_atoms.pbc, dtype=bool)):
            raise RuntimeError(f"{label}/{source}: PBC mismatch.")
        dft_aligned.append(dft_atoms)
        short_aligned.append(short_atoms)
    return dft_aligned, short_aligned

def require_file(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)

def discover_datasets():
    datasets = []
    require_file(TEST_DFT)
    require_file(TEST_SHORT)
    datasets.append({
        "name": "independent_test",
        "dft": TEST_DFT,
        "short": TEST_SHORT,
    })

    p_round = PARENT_ROUND if "PARENT_ROUND" in globals() else cfg.PARENT_ROUND
    c_round = ROUND if "ROUND" in globals() else cfg.ROUND

    if p_round >= 0:
        round_train_dft = getattr(cfg, "ROUND_TRAIN_DFT", None)
        round_train_short = getattr(cfg, "ROUND_TRAIN_SHORT", None)
        if not round_train_dft or not round_train_short:
            raise RuntimeError(
                f"v{c_round}: ROUND_TRAIN_DFT and ROUND_TRAIN_SHORT are required."
            )
        require_file(round_train_dft)
        require_file(round_train_short)
        datasets.append({
            "name": f"v{c_round}_enrichment",
            "dft": round_train_dft,
            "short": round_train_short,
        })
        for generation in range(START_HOLDOUT_GEN, c_round + 1):
            md_root = getattr(cfg, "MD_ROOT", None)
            if md_root is None:
                raise RuntimeError("MD_ROOT is required for historical holdout evaluation.")
            base = md_root / f"v{generation}"
            dft = base / f"v{generation}_holdout_dft.extxyz"
            short = base / f"v{generation}_holdout_short.extxyz"
            require_file(dft)
            require_file(short)
            datasets.append({
                "name": f"v{generation}_holdout",
                "dft": dft,
                "short": short,
            })
    else:
        out_valid_dft = getattr(cfg, "OUT_VALID_DFT", None)
        out_valid_short = getattr(cfg, "OUT_VALID_SHORT", None)
        if out_valid_dft and out_valid_short and out_valid_dft.is_file() and out_valid_short.is_file() and out_valid_dft.stat().st_size > 0:
            datasets.append({
                "name": "validation",
                "dft": out_valid_dft,
                "short": out_valid_short,
            })
    return datasets


def find_model(generation, branch):
    if branch == "full":
        train_dir = cfg.TRAINED_MODELS_ROOT / f"train_full_v{generation}"
        model_name = f"MACE_full_v{generation}"
    elif branch == "short":
        train_dir = cfg.TRAINED_MODELS_ROOT / f"train_short_v{generation}"
        model_name = f"MACE_short_v{generation}"
    else:
        raise ValueError(branch)
    checkpoint_dir = train_dir / "checkpoints"
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(checkpoint_dir)
    matches = sorted(
        path for path in checkpoint_dir.glob(f"{model_name}_run-*.model")
        if "stagetwo" not in path.name.lower()
    )
    if len(matches) != 1:
        candidates = sorted(
            path for path in [checkpoint_dir / f"{model_name}.model", train_dir / f"{model_name}.model"]
            if path.is_file() and "stagetwo" not in path.name.lower() and "compiled" not in path.name.lower()
        )
        if len(candidates) >= 1:
            return candidates[0]
        raise RuntimeError(f"Expected one {branch} model for v{generation}, found {len(matches)}:\n" + "\n".join(str(x) for x in matches))
    return matches[0]

def predict(frames, calculator):
    energies, forces = [], []
    for atoms in frames:
        a = atoms.copy()
        a.calc = calculator
        energies.append(float(a.get_potential_energy()))
        forces.append(np.asarray(a.get_forces(), dtype=np.float64))
    return np.asarray(energies, dtype=np.float64), forces

def release_calculator(calculator):
    del calculator
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def force_frame_metrics(ref, pred):
    ref = np.asarray(ref, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    delta = pred - ref
    component_rmse = float(np.sqrt(np.mean(delta ** 2)))
    component_mae = float(np.mean(np.abs(delta)))
    atom_vector_error = np.linalg.norm(delta, axis=1)
    worst_atom = int(np.argmax(atom_vector_error))
    max_atom_error = float(atom_vector_error[worst_atom])
    p95_atom_error = float(np.percentile(atom_vector_error, 95.0))
    return {
        "component_rmse": component_rmse,
        "component_mae": component_mae,
        "p95_atom_error": p95_atom_error,
        "max_atom_error": max_atom_error,
        "worst_atom": worst_atom,
    }

def is_in_sample(dataset_name, generation):
    return dataset_name == f"v{ROUND}_enrichment" and generation == ROUND

def build_summary_row(
    generation, dataset_name, system, indices, natoms,
    dft_energy, full_energy, total_energy, dft_forces, full_forces, total_forces, in_sample,
):
    indices = list(indices)
    e_ref_pa = np.asarray([dft_energy[i] / natoms[i] for i in indices])
    e_full_pa = np.asarray([full_energy[i] / natoms[i] for i in indices])
    e_total_pa = np.asarray([total_energy[i] / natoms[i] for i in indices])
    f_ref = flatten_forces([dft_forces[i] for i in indices])
    f_full = flatten_forces([full_forces[i] for i in indices])
    f_total = flatten_forces([total_forces[i] for i in indices])
    full_max, total_max = 0.0, 0.0
    for i in indices:
        full_max = max(full_max, float(np.max(np.linalg.norm(full_forces[i] - dft_forces[i], axis=1))))
        total_max = max(total_max, float(np.max(np.linalg.norm(total_forces[i] - dft_forces[i], axis=1))))
    return {
        "generation": f"v{generation}",
        "dataset": dataset_name,
        "in_sample": bool(in_sample),
        "system": system,
        "frames": len(indices),
        "E_RMSE_full_meV_atom": 1000.0 * rmse(e_ref_pa, e_full_pa),
        "E_RMSE_shortQEq_meV_atom": 1000.0 * rmse(e_ref_pa, e_total_pa),
        "E_MAE_full_meV_atom": 1000.0 * mae(e_ref_pa, e_full_pa),
        "E_MAE_shortQEq_meV_atom": 1000.0 * mae(e_ref_pa, e_total_pa),
        "F_RMSE_full_eV_A": rmse(f_ref, f_full),
        "F_RMSE_shortQEq_eV_A": rmse(f_ref, f_total),
        "F_MAE_full_eV_A": mae(f_ref, f_full),
        "F_MAE_shortQEq_eV_A": mae(f_ref, f_total),
        "F_maxerr_full_eV_A": full_max,
        "F_maxerr_shortQEq_eV_A": total_max,
    }

def element_rows(generation, dataset_name, system, dft_frames, dft_forces, full_forces, total_forces):
    rows = []
    elements = sorted({s for atoms in dft_frames for s in atoms.get_chemical_symbols()})
    for element in elements:
        ref_components, full_components, total_components = [], [], []
        full_vectors, total_vectors = [], []
        natoms_element = 0
        for atoms, ref, full, total in zip(dft_frames, dft_forces, full_forces, total_forces):
            symbols = np.asarray(atoms.get_chemical_symbols())
            mask = symbols == element
            if not np.any(mask):
                continue
            natoms_element += int(np.sum(mask))
            ref_e = np.asarray(ref)[mask]
            full_e = np.asarray(full)[mask]
            total_e = np.asarray(total)[mask]
            ref_components.append(ref_e.reshape(-1))
            full_components.append(full_e.reshape(-1))
            total_components.append(total_e.reshape(-1))
            full_vectors.extend(np.linalg.norm(full_e - ref_e, axis=1))
            total_vectors.extend(np.linalg.norm(total_e - ref_e, axis=1))
        ref_components = np.concatenate(ref_components)
        full_components = np.concatenate(full_components)
        total_components = np.concatenate(total_components)
        full_vectors = np.asarray(full_vectors, dtype=np.float64)
        total_vectors = np.asarray(total_vectors, dtype=np.float64)
        rows.append({
            "generation": f"v{generation}",
            "dataset": dataset_name,
            "system": system,
            "element": element,
            "atoms": natoms_element,
            "F_RMSE_full_eV_A": rmse(ref_components, full_components),
            "F_RMSE_shortQEq_eV_A": rmse(ref_components, total_components),
            "F_MAE_full_eV_A": mae(ref_components, full_components),
            "F_MAE_shortQEq_eV_A": mae(ref_components, total_components),
            "F_p95_atomerr_full_eV_A": float(np.percentile(full_vectors, 95.0)),
            "F_p95_atomerr_shortQEq_eV_A": float(np.percentile(total_vectors, 95.0)),
            "F_max_atomerr_full_eV_A": float(np.max(full_vectors)),
            "F_max_atomerr_shortQEq_eV_A": float(np.max(total_vectors)),
        })
    return rows

def evaluate_dataset(generation, dataset_name, dft_path, short_path, calc_full, calc_short):
    dft_frames = read(dft_path, index=":")
    short_frames = read(short_path, index=":")
    dft_frames, short_frames = align_datasets(dft_frames, short_frames, dataset_name)
    if not dft_frames:
        raise RuntimeError(f"{dataset_name}: empty dataset.")
    systems = [get_system(atoms) for atoms in dft_frames]
    natoms = np.asarray([len(atoms) for atoms in dft_frames], dtype=np.int64)

    dft_energy = np.asarray([get_dft_energy(a) for a in dft_frames], dtype=np.float64)
    dft_forces = [get_dft_forces(a) for a in dft_frames]
    short_energy = np.asarray([get_short_energy(a) for a in short_frames], dtype=np.float64)
    short_forces = [get_short_forces(a) for a in short_frames]
    qeq_energy = np.asarray([get_qeq_energy(a) for a in short_frames], dtype=np.float64)
    qeq_forces = [get_qeq_forces(a) for a in short_frames]

    for name, values in (
        ("DFT energy", dft_energy),
        ("Short energy", short_energy),
        ("QEq energy", qeq_energy),
    ):
        if not np.all(np.isfinite(values)):
            raise RuntimeError(f"{dataset_name}: non-finite {name}.")
    for name, values in (
        ("DFT forces", dft_forces),
        ("Short forces", short_forces),
        ("QEq forces", qeq_forces),
    ):
        if any(not np.all(np.isfinite(f)) for f in values):
            raise RuntimeError(f"{dataset_name}: non-finite {name}.")

    reconstructed_energy = short_energy + qeq_energy
    reconstructed_forces = [fs + fq for fs, fq in zip(short_forces, qeq_forces)]
    decomp_e = 1000.0 * rmse(reconstructed_energy / natoms, dft_energy / natoms)
    decomp_f = rmse(flatten_forces(reconstructed_forces), flatten_forces(dft_forces))
    if decomp_e > DECOMP_E_TOL or decomp_f > DECOMP_F_TOL:
        raise RuntimeError(f"{dataset_name}: DFT != short + QEq consistency failure.")

    full_energy, full_forces = predict(dft_frames, calc_full)
    pred_short_energy, pred_short_forces = predict(short_frames, calc_short)
    for name, values in (
        ("Full prediction energy", full_energy),
        ("Short prediction energy", pred_short_energy),
    ):
        if not np.all(np.isfinite(values)):
            raise RuntimeError(f"{dataset_name}: non-finite {name}.")
    for name, values in (
        ("Full prediction forces", full_forces),
        ("Short prediction forces", pred_short_forces),
    ):
        if any(not np.all(np.isfinite(f)) for f in values):
            raise RuntimeError(f"{dataset_name}: non-finite {name}.")

    total_energy = pred_short_energy + qeq_energy
    total_forces = [fm + fq for fm, fq in zip(pred_short_forces, qeq_forces)]

    summary_rows = [
        build_summary_row(
            generation, dataset_name, "ALL", range(len(dft_frames)), natoms,
            dft_energy, full_energy, total_energy, dft_forces, full_forces, total_forces,
            is_in_sample(dataset_name, generation),
        )
    ]
    for sys_name in sorted(set(systems)):
        idx = [i for i, v in enumerate(systems) if v == sys_name]
        summary_rows.append(
            build_summary_row(
                generation, dataset_name, sys_name, idx, natoms,
                dft_energy, full_energy, total_energy, dft_forces, full_forces, total_forces,
                is_in_sample(dataset_name, generation),
            )
        )

    frame_rows, worst_rows = [], []
    for i, (atoms, s_name, ref_e, pred_f_e, pred_t_e, ref_f, pred_f_f, pred_t_f) in enumerate(
        zip(dft_frames, systems, dft_energy, full_energy, total_energy, dft_forces, full_forces, total_forces)
    ):
        source = atoms.info.get("source", f"frame_{i}")
        n = len(atoms)
        ff = force_frame_metrics(ref_f, pred_f_f)
        tf = force_frame_metrics(ref_f, pred_t_f)
        symbols = atoms.get_chemical_symbols()
        row = {
            "generation": f"v{generation}",
            "dataset": dataset_name,
            "frame_index": i,
            "source": source,
            "system": s_name,
            "natoms": n,
            "E_error_full_meV_atom": 1000.0 * (pred_f_e - ref_e) / n,
            "E_abs_error_full_meV_atom": abs(1000.0 * (pred_f_e - ref_e) / n),
            "E_error_shortQEq_meV_atom": 1000.0 * (pred_t_e - ref_e) / n,
            "E_abs_error_shortQEq_meV_atom": abs(1000.0 * (pred_t_e - ref_e) / n),
            "F_RMSE_full_eV_A": ff["component_rmse"],
            "F_RMSE_shortQEq_eV_A": tf["component_rmse"],
            "F_p95_atomerr_full_eV_A": ff["p95_atom_error"],
            "F_p95_atomerr_shortQEq_eV_A": tf["p95_atom_error"],
            "F_max_atomerr_full_eV_A": ff["max_atom_error"],
            "F_max_atomerr_shortQEq_eV_A": tf["max_atom_error"],
            "worst_atom_full": ff["worst_atom"],
            "worst_element_full": symbols[ff["worst_atom"]],
            "worst_atom_shortQEq": tf["worst_atom"],
            "worst_element_shortQEq": symbols[tf["worst_atom"]],
        }
        frame_rows.append(row)
        worst_rows.append({
            "generation": f"v{generation}",
            "dataset": dataset_name,
            "system": s_name,
            "frame_index": i,
            "source": source,
            "branch": "full",
            "E_abs_error_meV_atom": row["E_abs_error_full_meV_atom"],
            "F_RMSE_eV_A": row["F_RMSE_full_eV_A"],
            "F_max_atomerr_eV_A": row["F_max_atomerr_full_eV_A"],
            "worst_atom": row["worst_atom_full"],
            "worst_element": row["worst_element_full"],
        })
        worst_rows.append({
            "generation": f"v{generation}",
            "dataset": dataset_name,
            "system": s_name,
            "frame_index": i,
            "source": source,
            "branch": "shortQEq",
            "E_abs_error_meV_atom": row["E_abs_error_shortQEq_meV_atom"],
            "F_RMSE_eV_A": row["F_RMSE_shortQEq_eV_A"],
            "F_max_atomerr_eV_A": row["F_max_atomerr_shortQEq_eV_A"],
            "worst_atom": row["worst_atom_shortQEq"],
            "worst_element": row["worst_element_shortQEq"],
        })

    elements = element_rows(generation, dataset_name, "ALL", dft_frames, dft_forces, full_forces, total_forces)
    for sys_name in sorted(set(systems)):
        idx = [i for i, name in enumerate(systems) if name == sys_name]
        elements.extend(element_rows(
            generation, dataset_name, sys_name,
            [dft_frames[i] for i in idx], [dft_forces[i] for i in idx],
            [full_forces[i] for i in idx], [total_forces[i] for i in idx],
        ))
    return summary_rows, frame_rows, elements, worst_rows

def build_change_table(summary):
    if (PARENT_ROUND if 'PARENT_ROUND' in globals() else cfg.PARENT_ROUND) < 0:
        return pd.DataFrame()

    rows = []
    for (dataset, sys_name), group in summary.groupby(["dataset", "system"], sort=False):
        parent = group[group["generation"] == f"v{PARENT_ROUND}"]
        current = group[group["generation"] == f"v{ROUND}"]
        if len(parent) != 1 or len(current) != 1:
            continue
        p = parent.iloc[0]
        c = current.iloc[0]
        row = {"dataset": dataset, "system": sys_name, "frames": int(c["frames"])}
        for metric in CHANGE_METRICS:
            old = float(p[metric])
            new = float(c[metric])
            row[f"{metric}_parent"] = old
            row[f"{metric}_current"] = new
            row[f"{metric}_delta"] = new - old
            row[f"{metric}_change_pct"] = (100.0 * (new - old) / old) if old != 0.0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)

def main():
    cfg.validate_config()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    datasets = discover_datasets()

    summary_rows, frame_rows, element_metric_rows, worst_rows = [], [], [], []

    for generation in ([ROUND] if PARENT_ROUND < 0 else [PARENT_ROUND, ROUND]):
        full_model = find_model(generation, "full")
        short_model = find_model(generation, "short")
        calc_full = MACECalculator(model_paths=[str(full_model)], device=DEVICE)
        calc_short = MACECalculator(model_paths=[str(short_model)], device=DEVICE)

        for ds in datasets:
            ds_sum, ds_f, ds_elem, ds_w = evaluate_dataset(
                generation=generation,
                dataset_name=ds["name"],
                dft_path=ds["dft"],
                short_path=ds["short"],
                calc_full=calc_full,
                calc_short=calc_short,
            )
            summary_rows.extend(ds_sum)
            frame_rows.extend(ds_f)
            element_metric_rows.extend(ds_elem)
            worst_rows.extend(ds_w)

        release_calculator(calc_full)
        release_calculator(calc_short)

    summary = pd.DataFrame(summary_rows)
    frames = pd.DataFrame(frame_rows)
    elements = pd.DataFrame(element_metric_rows)
    worst = pd.DataFrame(worst_rows).sort_values(
        ["F_max_atomerr_eV_A", "F_RMSE_eV_A"], ascending=[False, False]
    ).reset_index(drop=True)
    changes = build_change_table(summary)

    summary.to_csv(SUMMARY_FILE, index=False)
    frames.to_csv(FRAME_FILE, index=False)
    elements.to_csv(ELEMENT_FILE, index=False)
    worst.to_csv(WORST_FILE, index=False)
    changes.to_csv(CHANGE_FILE, index=False)

    print(f"EVALUATION COMPLETE: v{PARENT_ROUND} -> v{ROUND}\nSummary: {SUMMARY_FILE}\nChanges: {CHANGE_FILE}")

if __name__ == "__main__":
    main()
