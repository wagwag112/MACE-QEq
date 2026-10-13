#!/usr/bin/env python3
from __future__ import annotations
import argparse
import csv
import math
import time
from pathlib import Path
import sys
SCRIPT_DIR = Path(__file__).resolve().parent
AL_DIR = SCRIPT_DIR.parent
if str(AL_DIR) not in sys.path:
    sys.path.insert(0, str(AL_DIR))
try:
    import al_config as cfg
except Exception:
    cfg = None

import numpy as np
import os
try:
    import torch
    if not torch.cuda.is_available():
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
except Exception:
    pass
import jax
import jax.numpy as jnp
import jaxopt
from jax import value_and_grad
from jax.scipy.special import erfc
from ase.io import read, write
from ase.neighborlist import neighbor_list
from dmff.utils import pair_buffer_scales, regularize_pairs
from dmff.admp.recip import generate_pme_recip, Ck_1
from dmff.admp.pme import energy_pme

jax.config.update("jax_enable_x64", True)

KJ_MOL_PER_EV = 96.4869
COULOMB_KJ_MOL_ANG_E2 = 1389.35455846
PME_KAPPA = 4.3804348
PME_ORDER = 6
REQUESTED_RC = 6.0
RC_SAFETY = 0.499
TOL = 1.0e-8
MAXITER = 1000
PME_POINTS_PER_ANG = 1.2
PME_GRID_BLOCK = 8
PME_GRID_MIN = 16
PAIR_CAPACITY_FACTOR = 1.2
MAX_PAIRS = 0
DEFAULT_OUTPUT_EXTXYZ = "dataset_short.extxyz"
DEFAULT_REPORT_CSV = "qeq_report.csv"

R_COVALENCE = (
    0.31, 0.28, 1.28, 0.96, 0.84, 0.76, 0.71, 0.66, 0.57, 0.58,
    1.66, 1.41, 1.21, 1.11, 1.07, 1.05, 1.02, 1.06, 2.03, 1.76,
    1.70, 1.60, 1.53, 1.39, 1.61, 1.52, 1.50, 1.24, 1.32, 1.22,
    1.22, 1.20, 1.19, 1.20, 1.20, 1.16, 2.20, 1.95, 1.90, 1.75,
    1.64, 1.54, 1.47, 1.46, 1.42, 1.39, 1.45, 1.44, 1.42, 1.39,
    1.39, 1.38, 1.39, 1.40, 2.44, 2.15, 2.07, 2.04, 2.03, 2.01,
    1.99, 1.98, 1.98, 1.96, 1.94, 1.92, 1.92, 1.89, 1.90, 1.87,
    1.87, 1.75, 1.70, 1.62, 1.51, 1.44, 1.41, 1.36, 1.36, 1.32,
    1.45, 1.46, 1.48, 1.40, 1.50, 1.50, 2.60, 2.21, 2.15, 2.06,
    2.00, 1.96, 1.90, 1.87, 1.80, 1.69
)

CHI_HARDNESS_TABLE = getattr(cfg, "CHI_HARDNESS_TABLE", {
    "Li": (-3.0000, 10.0241),
    "N": (7.6100, 7.7500),
    "P": (1.8000, 7.0946),
    "S": (6.5745, 9.0000),
    "Cl": (9.9704, 6.1703),
})

def cell_heights(box):
    box = np.asarray(box, dtype=np.float64)
    volume = abs(np.linalg.det(box))
    if volume <= 0.0:
        raise RuntimeError("Non-positive cell volume.")
    ha = volume / np.linalg.norm(np.cross(box[1], box[2]))
    hb = volume / np.linalg.norm(np.cross(box[2], box[0]))
    hc = volume / np.linalg.norm(np.cross(box[0], box[1]))
    return np.asarray([ha, hb, hc], dtype=np.float64)

def safe_cutoff(box):
    heights = cell_heights(box)
    rc_max = RC_SAFETY * float(np.min(heights))
    rc = min(REQUESTED_RC, rc_max)
    if rc <= 0.0:
        raise RuntimeError(f"Invalid QEq cutoff: {rc}")
    return float(rc), heights

def round_grid(x):
    return max(PME_GRID_MIN, int(math.ceil(x / PME_GRID_BLOCK) * PME_GRID_BLOCK))

def automatic_pme_mesh(atoms):
    lengths = np.asarray(atoms.cell.lengths(), dtype=np.float64)
    return tuple(round_grid(length * PME_POINTS_PER_ANG) for length in lengths)

def get_neighbor_pairs(atoms, rc):
    i, j = neighbor_list("ij", atoms, cutoff=rc, self_interaction=False)
    mask = i < j
    i, j = i[mask], j[mask]
    if len(i) == 0:
        raise RuntimeError("Neighbor list is empty.")
    pairs = np.zeros((len(i), 3), dtype=np.int32)
    pairs[:, 0] = i
    pairs[:, 1] = j
    return jnp.asarray(pairs, dtype=jnp.int32)

def pair_distances(positions, box, pairs):
    pos1 = positions[pairs[:, 0]]
    pos2 = positions[pairs[:, 1]]
    box_inv = jnp.linalg.inv(box)
    dfrac = (pos1 - pos2) @ box_inv
    dfrac -= jnp.floor(dfrac + 0.5)
    dr = dfrac @ box
    return jnp.linalg.norm(dr, axis=1)

def make_qeq_energy(mesh, include_dipole, dipole_axis):
    K1, K2, K3 = mesh
    pme_recip_fn = generate_pme_recip(
        Ck_fn=Ck_1,
        kappa=PME_KAPPA / 10.0,
        gamma=False,
        pme_order=PME_ORDER,
        K1=K1,
        K2=K2,
        K3=K3,
        lmax=0,
    )
    mscales = jnp.asarray([1.0] * 6, dtype=jnp.float64)

    def qeq_energy(charges, positions, box, pairs, sigma, chi, hardness):
        charge_column = charges.reshape((-1, 1))
        e_pme = energy_pme(
            positions,
            box,
            pairs,
            charge_column,
            None,
            None,
            None,
            mscales,
            None,
            None,
            None,
            pme_recip_fn,
            PME_KAPPA / 10.0,
            K1,
            K2,
            K3,
            0,
            False,
        )
        ds = pair_distances(positions, box, pairs)
        scales = pair_buffer_scales(pairs)
        i = pairs[:, 0]
        j = pairs[:, 1]
        sigma_ij = jnp.sqrt(sigma[i] ** 2 + sigma[j] ** 2)
        e_pair = charges[i] * charges[j] * erfc(ds / (jnp.sqrt(2.0) * sigma_ij)) * COULOMB_KJ_MOL_ANG_E2 / ds * scales
        e_self = charges ** 2 * COULOMB_KJ_MOL_ANG_E2 / (2.0 * jnp.sqrt(jnp.pi) * sigma)
        e_gaussian = -jnp.sum(e_pair) + jnp.sum(e_self)
        e_onsite = jnp.sum((chi * charges + 0.5 * hardness * charges ** 2) * KJ_MOL_PER_EV)
        if include_dipole:
            volume = jnp.abs(jnp.linalg.det(box))
            dipole = jnp.sum(charges * positions[:, dipole_axis])
            e_dipole = 2.0 * jnp.pi / volume * COULOMB_KJ_MOL_ANG_E2 * dipole ** 2
        else:
            e_dipole = 0.0
        return (e_pme + e_gaussian + e_onsite + e_dipole) / KJ_MOL_PER_EV

    return qeq_energy

def make_projected_fun(energy_fn):
    def projected(charges, positions, box, pairs, sigma, chi, hardness):
        energy, grad_q = jax.value_and_grad(energy_fn)(charges, positions, box, pairs, sigma, chi, hardness)
        grad_q = grad_q - jnp.mean(grad_q)
        return energy, grad_q
    return projected

class FixedNeighborList:
    def __init__(self, atoms, rcut, capacity_factor=PAIR_CAPACITY_FACTOR, max_pairs=MAX_PAIRS):
        self.rcut = float(rcut)
        self.natoms = len(atoms)
        initial_pairs = self.real_pairs(atoms)
        n_initial = len(initial_pairs)
        if n_initial == 0:
            raise RuntimeError("Initial neighbor list is empty.")
        if max_pairs > 0:
            self.capacity = int(max_pairs)
        else:
            self.capacity = max(int(np.ceil(n_initial * float(capacity_factor))), n_initial + 1)
        if self.capacity < n_initial:
            raise RuntimeError(f"Initial pair count {n_initial} exceeds capacity {self.capacity}.")

    def real_pairs(self, atoms):
        pairs = np.asarray(get_neighbor_pairs(atoms, self.rcut), dtype=np.int32)
        if pairs.ndim != 2 or pairs.shape[1] != 3:
            raise RuntimeError(f"Unexpected pair shape: {pairs.shape}")
        return pairs

    def build(self, atoms):
        real_pairs = self.real_pairs(atoms)
        n_real = len(real_pairs)
        if n_real > self.capacity:
            raise RuntimeError(f"Neighbor-list capacity exceeded: {n_real} > {self.capacity}.")
        pairs = real_pairs.copy()
        n_padding = self.capacity - n_real
        if n_padding > 0:
            padding = np.zeros((n_padding, 3), dtype=np.int32)
            padding[:, 0] = self.natoms
            padding[:, 1] = self.natoms
            pairs = np.vstack((pairs, padding))
        pairs = jnp.asarray(pairs, dtype=jnp.int32)
        pairs = pairs.at[:, :2].set(regularize_pairs(pairs[:, :2]))
        if pairs.shape[0] != self.capacity:
            raise RuntimeError("Fixed pair shape corrupted.")
        return pairs, n_real

class QEqSolverFast:
    def __init__(
        self,
        atoms,
        mesh,
        include_dipole,
        dipole_axis=2,
        total_charge=0.0,
        tol=TOL,
        maxiter=MAXITER,
        capacity_factor=PAIR_CAPACITY_FACTOR,
        max_pairs=MAX_PAIRS,
    ):
        self.mesh = tuple(int(x) for x in mesh)
        self.include_dipole = bool(include_dipole)
        self.dipole_axis = int(dipole_axis)
        self.total_charge = float(total_charge)
        self.natoms = len(atoms)
        self.symbols = tuple(atoms.get_chemical_symbols())
        rc, _ = safe_cutoff(atoms.cell.array)
        self.rc = float(rc)
        self.nbl = FixedNeighborList(atoms=atoms, rcut=self.rc, capacity_factor=capacity_factor, max_pairs=max_pairs)
        missing = set(self.symbols) - set(CHI_HARDNESS_TABLE)
        if missing:
            raise RuntimeError("Missing (chi, hardness) parameters for: " + str(sorted(missing)))
        atomic_numbers = atoms.get_atomic_numbers()
        for z in atomic_numbers:
            if z < 1 or z > len(R_COVALENCE):
                raise RuntimeError(f"Atomic number {z} out of R_COVALENCE range (1-{len(R_COVALENCE)})")
        self.sigma = jnp.asarray([R_COVALENCE[z - 1] for z in atomic_numbers], dtype=jnp.float64)
        self.chi = jnp.asarray([CHI_HARDNESS_TABLE[s][0] for s in self.symbols], dtype=jnp.float64)
        self.hardness = jnp.asarray([CHI_HARDNESS_TABLE[s][1] for s in self.symbols], dtype=jnp.float64)
        self.energy_fn = make_qeq_energy(mesh=self.mesh, include_dipole=self.include_dipole, dipole_axis=self.dipole_axis)
        projected = make_projected_fun(self.energy_fn)
        self.optimizer = jaxopt.LBFGS(fun=projected, value_and_grad=True, tol=tol, maxiter=maxiter, jit=True, implicit_diff=False)
        self.energy_force_fn = jax.jit(value_and_grad(self.energy_fn, argnums=1))
        self.previous_q = None

    def initial_charge(self):
        if self.previous_q is not None and len(self.previous_q) == self.natoms:
            q0 = self.previous_q.copy()
            q0 += (self.total_charge - q0.sum()) / self.natoms
            return jnp.asarray(q0, dtype=jnp.float64)
        return jnp.full(self.natoms, self.total_charge / self.natoms, dtype=jnp.float64)

    def compute(self, atoms, chi=None):
        if len(atoms) != self.natoms:
            raise RuntimeError("Atom count changed.")
        if tuple(atoms.get_chemical_symbols()) != self.symbols:
            raise RuntimeError("Atomic species/order changed.")
        if not np.all(atoms.pbc):
            raise RuntimeError("QEq PME requires fully periodic structures.")
        positions = jnp.asarray(atoms.positions, dtype=jnp.float64)
        box = jnp.asarray(atoms.cell.array, dtype=jnp.float64)
        pairs, n_real = self.nbl.build(atoms)
        q0 = self.initial_charge()
        if chi is None:
            chi_use = self.chi
        else:
            chi_use = jnp.asarray(chi, dtype=jnp.float64)
            if chi_use.shape != self.chi.shape:
                raise RuntimeError(f"Custom chi has wrong shape: {chi_use.shape}, expected {self.chi.shape}")
        t0 = time.perf_counter()
        result = self.optimizer.run(q0, positions, box, pairs, self.sigma, chi_use, self.hardness)
        q_opt = result.params
        q_opt.block_until_ready()
        solve_s = time.perf_counter() - t0
        charges = np.asarray(q_opt, dtype=np.float64)
        iterations = int(result.state.iter_num)
        if not np.all(np.isfinite(charges)):
            raise RuntimeError("Non-finite optimized charges.")
        t0 = time.perf_counter()
        energy, grad_positions = self.energy_force_fn(q_opt, positions, box, pairs, self.sigma, chi_use, self.hardness)
        energy.block_until_ready()
        grad_positions.block_until_ready()
        force_s = time.perf_counter() - t0
        qeq_energy = float(np.asarray(energy))
        qeq_forces = -np.asarray(grad_positions, dtype=np.float64)
        if not np.isfinite(qeq_energy):
            raise RuntimeError("Non-finite QEq energy.")
        if not np.all(np.isfinite(qeq_forces)):
            raise RuntimeError("Non-finite QEq forces.")
        charge_error = float(charges.sum() - self.total_charge)
        self.previous_q = charges.copy()
        return {
            "energy": qeq_energy,
            "forces": qeq_forces,
            "charges": charges,
            "charge_error": charge_error,
            "iterations": iterations,
            "n_pairs": int(n_real),
            "pair_capacity": int(self.nbl.capacity),
            "rc": self.rc,
            "solve_s": float(solve_s),
            "force_s": float(force_s),
        }

def force_vector_rms(forces):
    return float(np.sqrt(np.mean(np.sum(forces ** 2, axis=1))))

def force_max(forces):
    return float(np.max(np.linalg.norm(forces, axis=1)))

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
            is_cand = is_metal & (cn_other == 0)
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

def get_dipole_axis(atoms, system):
    value = atoms.info.get("qeq_dipole_axis")
    if value is not None:
        axis = int(value)
        if axis in {0, 1, 2}:
            return axis
        if axis == -1:
            return None
        raise RuntimeError(f"Invalid qeq_dipole_axis: {axis}")
    if "_" in str(system):
        return 2
    return None

def build_residual_frame(atoms, dft_energy, dft_forces, result):
    e_qeq = float(result["energy"])
    f_qeq = np.asarray(result["forces"], dtype=np.float64)
    charges = np.asarray(result["charges"], dtype=np.float64)
    e_short = dft_energy - e_qeq
    f_short = dft_forces - f_qeq
    out = atoms.copy()
    out.calc = None
    out.info.pop("REF_energy", None)
    out.info.pop("REF_stress", None)
    if "REF_forces" in out.arrays:
        del out.arrays["REF_forces"]
    out.info["dft_energy"] = float(dft_energy)
    out.info["qeq_energy"] = float(e_qeq)
    out.info["short_energy"] = float(e_short)
    out.info["qeq_charge_error"] = float(result["charge_error"])
    out.info["qeq_rc"] = float(result["rc"])
    out.info["qeq_iterations"] = int(result["iterations"])
    out.info["qeq_pair_capacity"] = int(result["pair_capacity"])
    out.arrays["dft_forces"] = np.asarray(dft_forces, dtype=np.float64)
    out.arrays["qeq_forces"] = np.asarray(f_qeq, dtype=np.float64)
    out.arrays["qeq_charges"] = np.asarray(charges, dtype=np.float64)
    out.arrays["short_forces"] = np.asarray(f_short, dtype=np.float64)
    out.info["REF_energy"] = float(e_short)
    out.arrays["REF_forces"] = np.asarray(f_short, dtype=np.float64)
    return out

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT_EXTXYZ)
    parser.add_argument("--report", default=DEFAULT_REPORT_CSV)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=None)
    parser.add_argument("--pair-capacity-factor", type=float, default=PAIR_CAPACITY_FACTOR)
    parser.add_argument("--max-pairs", type=int, default=MAX_PAIRS)
    return parser.parse_args()

def main():
    args = parse_args()
    input_file = Path(args.input)
    output_file = Path(args.output)
    report_file = Path(args.report)
    if not input_file.exists():
        raise FileNotFoundError(input_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    all_frames = read(input_file, index=":")
    stop = len(all_frames) if args.stop is None else min(args.stop, len(all_frames))
    if args.start < 0 or stop <= args.start:
        raise ValueError("Invalid start/stop range.")
    frames = all_frames[args.start:stop]
    for i, atoms in enumerate(frames):
        if "REF_energy" not in atoms.info or "REF_forces" not in atoms.arrays:
            raise RuntimeError(f"Frame {i}: REF labels missing.")
    solver_cache = {}

    def get_solver(atoms, system):
        mesh = automatic_pme_mesh(atoms)
        dipole_axis = get_dipole_axis(atoms, system)
        include_dipole = dipole_axis is not None
        rc, _ = safe_cutoff(atoms.cell.array)
        key = (tuple(mesh), include_dipole, dipole_axis, round(float(rc), 4), len(atoms), tuple(atoms.get_chemical_symbols()))
        if key not in solver_cache:
            solver_cache[key] = QEqSolverFast(
                atoms=atoms,
                mesh=mesh,
                include_dipole=include_dipole,
                dipole_axis=dipole_axis if dipole_axis is not None else 2,
                total_charge=0.0,
                tol=TOL,
                maxiter=MAXITER,
                capacity_factor=args.pair_capacity_factor,
                max_pairs=args.max_pairs,
            )
        return solver_cache[key], mesh

    if output_file.exists():
        output_file.unlink()
    report = []

    for local_i, atoms in enumerate(frames):
        t0 = time.perf_counter()
        frame_index = args.start + local_i
        system = get_system(atoms)
        dipole_axis = get_dipole_axis(atoms, system)
        dft_energy = float(atoms.info["REF_energy"])
        dft_forces = np.asarray(atoms.arrays["REF_forces"], dtype=np.float64)
        solver, mesh = get_solver(atoms, system)
        result = solver.compute(atoms)
        qeq_energy = float(result["energy"])
        qeq_forces = np.asarray(result["forces"], dtype=np.float64)
        charges = np.asarray(result["charges"], dtype=np.float64)
        short_energy = dft_energy - qeq_energy
        short_forces = dft_forces - qeq_forces
        if not np.isfinite(short_energy) or not np.all(np.isfinite(short_forces)):
            raise RuntimeError(f"Frame {frame_index}: non-finite residual.")
        if abs(result["charge_error"]) > 1.0e-8:
            raise RuntimeError(f"Frame {frame_index}: charge error = {result['charge_error']:.3e} e")
        out = build_residual_frame(atoms, dft_energy, dft_forces, result)
        out.info["system"] = system
        write(output_file, out, format="extxyz", append=True)
        wall = time.perf_counter() - t0
        report.append({
            "frame": frame_index,
            "source": atoms.info.get("source", ""),
            "system": system,
            "natoms": len(atoms),
            "E_DFT_eV": dft_energy,
            "E_QEq_eV": qeq_energy,
            "E_short_eV": short_energy,
            "F_DFT_rms": force_vector_rms(dft_forces),
            "F_QEq_rms": force_vector_rms(qeq_forces),
            "F_short_rms": force_vector_rms(short_forces),
            "F_DFT_max": force_max(dft_forces),
            "F_QEq_max": force_max(qeq_forces),
            "F_short_max": force_max(short_forces),
            "max_abs_q": float(np.max(np.abs(charges))),
            "sum_q": float(charges.sum()),
            "iterations": result["iterations"],
            "pairs": result["n_pairs"],
            "pair_capacity": result["pair_capacity"],
            "rc_A": result["rc"],
            "mesh_K1": mesh[0],
            "mesh_K2": mesh[1],
            "mesh_K3": mesh[2],
            "dipole": (["x", "y", "z"][dipole_axis] if dipole_axis is not None else "OFF"),
            "qeq_solve_s": result["solve_s"],
            "qeq_force_s": result["force_s"],
            "wall_s": wall,
        })
        print(f"[{local_i + 1:4d}/{len(frames)}] {system:8s} frame={frame_index:4d} N={len(atoms):4d} E_QEq/N={qeq_energy/len(atoms):+.5f} F_QEq={report[-1]['F_QEq_rms']:.4f} F_short={report[-1]['F_short_rms']:.4f} iter={result['iterations']:4d} wall={wall:.2f}s")

    with open(report_file, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=report[0].keys())
        writer.writeheader()
        writer.writerows(report)

    output_frames = read(output_file, index=":")
    if len(output_frames) != len(frames):
        raise RuntimeError("Output frame count mismatch.")
    max_e_decomp = 0.0
    max_f_decomp = 0.0
    for i, (dft, short) in enumerate(zip(frames, output_frames)):
        e_ref_short = float(short.info["REF_energy"])
        e_short = float(short.info["short_energy"])
        f_ref_short = np.asarray(short.arrays["REF_forces"], dtype=np.float64)
        f_short = np.asarray(short.arrays["short_forces"], dtype=np.float64)
        if abs(e_ref_short - e_short) > 1.0e-12 or np.max(np.abs(f_ref_short - f_short)) > 1.0e-12:
            raise RuntimeError(f"Frame {i}: REF != short consistency error.")
        dft_e = float(short.info["dft_energy"])
        qeq_e = float(short.info["qeq_energy"])
        dft_f = np.asarray(short.arrays["dft_forces"], dtype=np.float64)
        qeq_f = np.asarray(short.arrays["qeq_forces"], dtype=np.float64)
        e_decomp = abs(dft_e - (e_short + qeq_e))
        f_decomp = float(np.max(np.abs(dft_f - (f_short + qeq_f))))
        max_e_decomp = max(max_e_decomp, e_decomp)
        max_f_decomp = max(max_f_decomp, f_decomp)

    print(f"SUMMARY: frames={len(report)} max_dE={max_e_decomp:.3e} eV max_dF={max_f_decomp:.3e} eV/A (Decomposition check PASS)")

if __name__ == "__main__":
    main()
