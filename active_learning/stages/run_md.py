#!/usr/bin/env python3
from __future__ import annotations
import argparse
import csv
import sys
import time
from pathlib import Path
import numpy as np
import jax
from ase import units
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.constraints import FixAtoms
from ase.neighborlist import neighbor_list
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.npt import NPT
from ase.calculators.calculator import Calculator, all_changes
from mace.calculators import MACECalculator

SCRIPT_DIR = Path(__file__).resolve().parent
AL_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = AL_DIR.parent
for p in [SCRIPT_DIR, AL_DIR, PROJECT_ROOT]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

DEFAULT_STRUCTURE = PROJECT_ROOT / "structures" / "Li_Li3N_tv.vasp"
DEFAULT_MODEL = PROJECT_ROOT / "trained_models" / "train_short_v0" / "checkpoints" / "MACE_short_v0.model"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "debug_output"

CONSTP_ANION_ELEMENT = "N"
CONSTP_RCUT_ANION = 2.5
CONSTP_RCUT_LILI = 3.5
CONSTP_FIXED_PLANES_PER_SIDE = 3
CONSTP_FIXED_PLANE_TOL = 0.35

DEFAULT_BOTTOM_EXTERNAL_CHI = 0.0
DEFAULT_UPPER_EXTERNAL_CHI = 0.0
DEFAULT_TEMPERATURE_K = 300.0
DEFAULT_MD_STEPS = 5000
DEFAULT_TIMESTEP_FS = 1.0
DEFAULT_TTIME_FS = 20.0
DEFAULT_SAVE_EVERY = 100
DEFAULT_SEED = 2333
DEFAULT_DEVICE = "cuda"

PRODUCTION_QEQ_TOL = 1.0e-4
PRODUCTION_QEQ_MAXITER = 100
PRODUCTION_PME_POINTS_PER_ANG = 1.2
PRODUCTION_PME_GRID_BLOCK = 4
PRODUCTION_PME_GRID_MIN = 16
DEFAULT_PAIR_CAPACITY_FACTOR = 1.15
DEFAULT_MAX_PAIRS = 0
DEFAULT_QEQ_UPDATE_EVERY = 3
DEFAULT_CN_CADENCE = 10

jax.config.update("jax_enable_x64", True)

try:
    import al_config as cfg
except Exception:
    cfg = None

def parse_anion_cutoffs(val) -> dict[str, float]:
    if val is None:
        if cfg is not None and hasattr(cfg, "ANION_CUTOFFS"):
            return {str(k): float(v) for k, v in cfg.ANION_CUTOFFS.items()}
        return {"N": 2.50, "P": 2.80, "S": 2.90, "Cl": 3.05}
    if isinstance(val, dict):
        return {str(k): float(v) for k, v in val.items()}
    val = str(val).strip()
    if val.startswith("{"):
        import json
        return {str(k): float(v) for k, v in json.loads(val).items()}
    res = {}
    for item in val.split(","):
        item = item.strip()
        if not item:
            continue
        k, v = item.split(":")
        res[k.strip()] = float(v.strip())
    return res

from active_learning.stages.build_qeq_residual import (
    QEqSolverFast,
    get_system,
    get_dipole_axis,
)

class FixAtomsAwareNPT(NPT):
    def _calculateconstants(self):
        super()._calculateconstants()
        fixed = set()
        for constraint in self.atoms.constraints:
            if isinstance(constraint, FixAtoms):
                fixed.update(int(i) for i in constraint.get_indices())
        n_total = len(self.atoms)
        n_fixed = len(fixed)
        n_mobile = n_total - n_fixed
        if n_mobile <= 1:
            raise RuntimeError(f"Invalid number of mobile atoms: {n_mobile}")
        if self.ttime is None:
            self.tfact = 0.0
        else:
            self.tfact = 2.0 / (3.0 * n_mobile * self.temperature * self.ttime * self.ttime)
        self.desiredEkin = 1.5 * (n_mobile if n_fixed > 0 else n_mobile - 1) * self.temperature

def production_pme_mesh(atoms, points_per_ang=PRODUCTION_PME_POINTS_PER_ANG):
    lengths = np.asarray(atoms.cell.lengths(), dtype=np.float64)
    def round_grid(x):
        return max(
            PRODUCTION_PME_GRID_MIN,
            int(np.ceil(x / PRODUCTION_PME_GRID_BLOCK) * PRODUCTION_PME_GRID_BLOCK),
        )
    return tuple(round_grid(L * points_per_ang) for L in lengths)

def identify_fixed_metal_planes(
    atoms,
    metal_symbol=None,
    planes_per_side=CONSTP_FIXED_PLANES_PER_SIDE,
    plane_tol=CONSTP_FIXED_PLANE_TOL,
):
    if planes_per_side <= 0 or plane_tol <= 0.0:
        raise ValueError("Invalid plane selection parameters.")
    metal_sym = str(metal_symbol or getattr(cfg, "ACTIVE_METAL", "Li"))
    symbols = np.asarray(atoms.get_chemical_symbols())
    z = np.asarray(atoms.positions[:, 2], dtype=np.float64)
    metal_indices = np.where(symbols == metal_sym)[0]
    if len(metal_indices) == 0:
        raise RuntimeError(f"No {metal_sym} atoms found.")
    z_mid = 0.5 * atoms.cell.lengths()[2]
    bottom_metal = metal_indices[z[metal_indices] < z_mid]
    upper_metal = metal_indices[z[metal_indices] > z_mid]
    if len(bottom_metal) == 0 or len(upper_metal) == 0:
        raise RuntimeError(f"Boundary {metal_sym} atoms missing on one or both sides.")

    def build_planes(indices, ascending):
        order = np.argsort(z[indices]) if ascending else np.argsort(z[indices])[::-1]
        sorted_indices = indices[order]
        planes = []
        current = [int(sorted_indices[0])]
        ref_z = float(z[sorted_indices[0]])
        for idx in sorted_indices[1:]:
            zi = float(z[idx])
            if abs(zi - ref_z) <= plane_tol:
                current.append(int(idx))
            else:
                planes.append(np.asarray(current, dtype=int))
                current = [int(idx)]
                ref_z = zi
        planes.append(np.asarray(current, dtype=int))
        return planes

    bottom_planes = build_planes(bottom_metal, ascending=True)
    upper_planes = build_planes(upper_metal, ascending=False)
    if len(bottom_planes) < planes_per_side or len(upper_planes) < planes_per_side:
        raise RuntimeError(f"Insufficient outer {metal_sym} planes detected.")

    fixed_bottom = np.sort(np.concatenate(bottom_planes[:planes_per_side]))
    fixed_upper = np.sort(np.concatenate(upper_planes[:planes_per_side]))
    fixed_all = np.sort(np.concatenate([fixed_bottom, fixed_upper]))

    if len(np.unique(fixed_all)) != len(fixed_all):
        raise RuntimeError(f"Duplicate indices in fixed {metal_sym} set.")
    if not np.all(symbols[fixed_all] == metal_sym):
        raise RuntimeError(f"Non-{metal_sym} atoms detected in mechanical fix constraint.")
    return fixed_bottom, fixed_upper, fixed_all


def determine_constp_chi(
    atoms,
    chi_base,
    bottom_seeds,
    upper_seeds,
    bottom_external_chi=0.0,
    upper_external_chi=0.0,
    active_metal=None,
    rcut_metal=None,
    anion_cutoffs=None,
    rcut_anion=CONSTP_RCUT_ANION,
    rcut_lili=CONSTP_RCUT_LILI,
    anion_symbol=CONSTP_ANION_ELEMENT,
):
    symbols = np.asarray(atoms.get_chemical_symbols())
    natoms = len(atoms)
    bottom_seeds = np.asarray(bottom_seeds, dtype=int)
    upper_seeds = np.asarray(upper_seeds, dtype=int)
    fixed_indices = np.sort(np.concatenate([bottom_seeds, upper_seeds]))

    metal_sym = str(active_metal or getattr(cfg, "ACTIVE_METAL", "Li"))
    r_metal = float(rcut_metal if rcut_metal is not None else (rcut_lili if rcut_lili is not None else getattr(cfg, "RCUT_METAL", 3.50)))
    if anion_cutoffs is not None:
        cutoffs = parse_anion_cutoffs(anion_cutoffs)
    elif cfg is not None and hasattr(cfg, "ANION_CUTOFFS"):
        cutoffs = {str(k): float(v) for k, v in cfg.ANION_CUTOFFS.items()}
    elif anion_symbol and rcut_anion:
        cutoffs = {str(anion_symbol): float(rcut_anion)}
    else:
        cutoffs = {"N": 2.50}

    max_rc = max(r_metal, max(cutoffs.values()) if cutoffs else 0.0)
    pair_i, pair_j, d = neighbor_list(
        "ijd",
        atoms,
        max_rc,
        self_interaction=False,
    )

    is_metal = (symbols == metal_sym)
    anion_bonds = np.zeros(len(pair_i), dtype=bool)
    for a_sym, a_rc in cutoffs.items():
        is_a = (symbols == a_sym)
        anion_bonds |= (is_metal[pair_i] & is_a[pair_j] & (d <= a_rc))

    cn_anion = np.bincount(pair_i[anion_bonds], minlength=natoms).astype(np.int32, copy=False)

    is_candidate = is_metal & (cn_anion == 0)
    is_candidate[bottom_seeds] = True
    is_candidate[upper_seeds] = True

    metal_bonds = is_candidate[pair_i] & is_candidate[pair_j] & (d <= r_metal)
    edges_i = pair_i[metal_bonds]
    edges_j = pair_j[metal_bonds]

    adj = csr_matrix(
        (np.ones(len(edges_i), dtype=bool), (edges_i, edges_j)),
        shape=(natoms, natoms),
    )
    n_components, component_labels = connected_components(
        csgraph=adj,
        directed=False,
        return_labels=True,
    )

    bottom_clusters = np.unique(component_labels[bottom_seeds])
    upper_clusters = np.unique(component_labels[upper_seeds])

    if np.any(np.isin(bottom_clusters, upper_clusters)):
        raise RuntimeError(f"Physical short-circuit: metallic {metal_sym} percolation path connects electrodes.")

    bottom_mask = np.isin(component_labels, bottom_clusters) & is_candidate
    upper_mask = np.isin(component_labels, upper_clusters) & is_candidate

    bottom_indices = np.where(bottom_mask)[0]
    upper_indices = np.where(upper_mask)[0]
    dead_metal_indices = np.where(is_candidate & (~bottom_mask) & (~upper_mask))[0]
    electrode_indices = np.sort(np.concatenate([bottom_indices, upper_indices]))

    chi_eff = np.asarray(chi_base, dtype=np.float64).copy()
    chi_eff[bottom_indices] += float(bottom_external_chi)
    chi_eff[upper_indices] += float(upper_external_chi)

    diagnostics = {
        "electrode_indices": electrode_indices,
        "bottom_indices": bottom_indices,
        "upper_indices": upper_indices,
        "dead_li_indices": dead_metal_indices,
        "dead_metal_indices": dead_metal_indices,
        "fixed_indices": fixed_indices,
        "n_components": int(n_components),
    }
    return chi_eff, diagnostics

class ElectrodeSwitchingTracker:
    def __init__(self, natoms):
        self.natoms = int(natoms)
        self.previous_mask = None
        self.n_evaluations = 0
        self.total_enter = 0
        self.total_leave = 0
        self.total_switch = 0
        self.enter_counts = np.zeros(self.natoms, dtype=np.int64)
        self.leave_counts = np.zeros(self.natoms, dtype=np.int64)

    def update(self, electrode_indices):
        current_mask = np.zeros(self.natoms, dtype=bool)
        current_mask[np.asarray(electrode_indices, dtype=int)] = True
        if self.previous_mask is None:
            entered = np.empty(0, dtype=int)
            left = np.empty(0, dtype=int)
        else:
            entered = np.where((~self.previous_mask) & current_mask)[0]
            left = np.where(self.previous_mask & (~current_mask))[0]
        n_enter = int(len(entered))
        n_leave = int(len(left))
        n_switch = int(n_enter + n_leave)
        self.n_evaluations += 1
        self.total_enter += n_enter
        self.total_leave += n_leave
        self.total_switch += n_switch
        if n_enter:
            self.enter_counts[entered] += 1
        if n_leave:
            self.leave_counts[left] += 1
        self.previous_mask = current_mask.copy()
        return {
            "entered_indices": entered.astype(int),
            "left_indices": left.astype(int),
            "n_enter": n_enter,
            "n_leave": n_leave,
            "n_switch": n_switch,
            "n_evaluations": int(self.n_evaluations),
            "total_enter": int(self.total_enter),
            "total_leave": int(self.total_leave),
            "total_switch": int(self.total_switch),
        }

class MACEQEqCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        model_path,
        atoms,
        mode="constp",
        fixed_indices=None,
        fixed_bottom_indices=None,
        fixed_upper_indices=None,
        device=DEFAULT_DEVICE,
        capacity_factor=DEFAULT_PAIR_CAPACITY_FACTOR,
        max_pairs=DEFAULT_MAX_PAIRS,
        bottom_external_chi=DEFAULT_BOTTOM_EXTERNAL_CHI,
        upper_external_chi=DEFAULT_UPPER_EXTERNAL_CHI,
        qeq_update_every=DEFAULT_QEQ_UPDATE_EVERY,
        qeq_tol=PRODUCTION_QEQ_TOL,
        pme_points_per_ang=PRODUCTION_PME_POINTS_PER_ANG,
        cn_cadence=DEFAULT_CN_CADENCE,
        active_metal=None,
        rcut_metal=None,
        anion_cutoffs=None,
        rcut_anion=CONSTP_RCUT_ANION,
        rcut_lili=CONSTP_RCUT_LILI,
        anion_symbol=CONSTP_ANION_ELEMENT,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mode = str(mode).lower()
        self.mace = MACECalculator(
            model_paths=[str(model_path)],
            device=device,
        )
        if self.mode == "full":
            self.qeq = None
            self.last_qeq = None
            self.last_constp_diagnostics = None
            self.electrode_tracker = None
            self.system = None
            self.dipole_axis = None
            self.initial_z = np.asarray(atoms.positions[:, 2], dtype=np.float64).copy()
            self.fixed_indices = (
                np.asarray(fixed_indices, dtype=int).copy()
                if fixed_indices is not None
                else np.array([], dtype=int)
            )
            return

        self.system = get_system(atoms)
        self.dipole_axis = get_dipole_axis(atoms, self.system)
        self.include_dipole = self.dipole_axis is not None
        self.pme_points_per_ang = float(pme_points_per_ang)
        self.mesh = production_pme_mesh(atoms, points_per_ang=self.pme_points_per_ang)
        self.qeq_tol = float(qeq_tol)
        self.qeq = QEqSolverFast(
            atoms=atoms,
            mesh=self.mesh,
            include_dipole=self.include_dipole,
            dipole_axis=self.dipole_axis if self.dipole_axis is not None else 2,
            total_charge=0.0,
            tol=self.qeq_tol,
            maxiter=PRODUCTION_QEQ_MAXITER,
            capacity_factor=capacity_factor,
            max_pairs=max_pairs,
        )
        self.cn_cadence = int(cn_cadence)
        self._cached_chi = None
        self._cached_constp_diag = None

        if self.mode == "constp":
            if fixed_bottom_indices is None or fixed_upper_indices is None:
                if fixed_indices is None or len(fixed_indices) == 0:
                    raise RuntimeError("ConstP calculator received no fixed electrode atoms.")
                z = atoms.positions[:, 2]
                z_mid = 0.5 * atoms.cell.lengths()[2]
                self.fixed_bottom_indices = fixed_indices[z[fixed_indices] < z_mid]
                self.fixed_upper_indices = fixed_indices[z[fixed_indices] > z_mid]
                self.fixed_indices = np.asarray(fixed_indices, dtype=int).copy()
            else:
                self.fixed_bottom_indices = np.asarray(fixed_bottom_indices, dtype=int).copy()
                self.fixed_upper_indices = np.asarray(fixed_upper_indices, dtype=int).copy()
                self.fixed_indices = np.sort(np.concatenate([self.fixed_bottom_indices, self.fixed_upper_indices]))

            if len(self.fixed_indices) == 0:
                raise RuntimeError("ConstP calculator received empty fixed electrode set.")
        else:
            self.fixed_bottom_indices = np.array([], dtype=int)
            self.fixed_upper_indices = np.array([], dtype=int)
            self.fixed_indices = np.asarray(fixed_indices, dtype=int).copy() if fixed_indices is not None else np.array([], dtype=int)

        self.bottom_external_chi = float(bottom_external_chi)
        self.upper_external_chi = float(upper_external_chi)
        self.qeq_update_every = int(qeq_update_every)
        if self.qeq_update_every <= 0:
            raise ValueError("qeq_update_every must be >= 1")

        self.active_metal = str(active_metal or getattr(cfg, "ACTIVE_METAL", "Li"))
        self.rcut_metal = float(rcut_metal if rcut_metal is not None else (rcut_lili if rcut_lili is not None else getattr(cfg, "RCUT_METAL", 3.50)))
        if anion_cutoffs is not None:
            self.anion_cutoffs = parse_anion_cutoffs(anion_cutoffs)
        elif cfg is not None and hasattr(cfg, "ANION_CUTOFFS"):
            self.anion_cutoffs = {str(k): float(v) for k, v in cfg.ANION_CUTOFFS.items()}
        elif anion_symbol and rcut_anion:
            self.anion_cutoffs = {str(anion_symbol): float(rcut_anion)}
        else:
            self.anion_cutoffs = {"N": 2.50}
        self.rcut_anion = float(rcut_anion)
        self.rcut_lili = self.rcut_metal
        self.anion_symbol = str(anion_symbol)
        self.qeq_eval_counter = 0
        self._cached_qeq = None
        self.last_qeq = None
        self.last_constp_diagnostics = None
        self.electrode_tracker = ElectrodeSwitchingTracker(len(atoms)) if self.mode == "constp" else None
        self.initial_z = np.asarray(atoms.positions[:, 2], dtype=np.float64).copy()

    def calculate(
        self,
        atoms=None,
        properties=("energy", "forces"),
        system_changes=all_changes,
    ):
        super().calculate(atoms, properties, system_changes)
        mace_atoms = atoms.copy()
        mace_atoms.calc = self.mace
        mace_energy = float(mace_atoms.get_potential_energy())
        mace_forces = np.asarray(mace_atoms.get_forces(), dtype=np.float64)

        if self.mode == "full":
            self.results["energy"] = mace_energy
            self.results["forces"] = mace_forces
            self.last_qeq = None
            self.last_constp_diagnostics = None
            return

        if self.mode == "constq" or (self.bottom_external_chi == 0.0 and self.upper_external_chi == 0.0):
            chi_eff = np.asarray(self.qeq.chi, dtype=np.float64)
            constp_diag = None
        else:
            cn_updated = (
                self._cached_chi is None
                or (self.qeq_eval_counter % self.cn_cadence == 0)
            )
            if cn_updated:
                chi_eff, constp_diag = determine_constp_chi(
                    atoms=atoms,
                    chi_base=np.asarray(self.qeq.chi, dtype=np.float64),
                    bottom_seeds=self.fixed_bottom_indices,
                    upper_seeds=self.fixed_upper_indices,
                    bottom_external_chi=self.bottom_external_chi,
                    upper_external_chi=self.upper_external_chi,
                    active_metal=self.active_metal,
                    rcut_metal=self.rcut_metal,
                    anion_cutoffs=self.anion_cutoffs,
                )
                if self.electrode_tracker is not None:
                    switch_diag = self.electrode_tracker.update(constp_diag["electrode_indices"])
                    constp_diag.update(switch_diag)
                self._cached_chi = chi_eff
                self._cached_constp_diag = constp_diag
            else:
                chi_eff = self._cached_chi
                constp_diag = dict(self._cached_constp_diag)

        qeq_updated = (
            self._cached_qeq is None
            or (self.qeq_eval_counter % self.qeq_update_every == 0)
        )

        if qeq_updated:
            qeq = self.qeq.compute(atoms, chi=chi_eff)
            self._cached_qeq = qeq
        else:
            qeq = dict(self._cached_qeq)

        qeq["qeq_updated"] = bool(qeq_updated)
        qeq["qeq_eval_counter"] = int(self.qeq_eval_counter)
        qeq_energy = float(qeq["energy"])
        qeq_forces = np.asarray(qeq["forces"], dtype=np.float64)
        self.qeq_eval_counter += 1

        self.results["energy"] = mace_energy + qeq_energy
        self.results["forces"] = mace_forces + qeq_forces
        self.last_qeq = qeq
        self.last_constp_diagnostics = constp_diag

def force_max(forces):
    forces = np.asarray(forces, dtype=np.float64)
    return float(np.max(np.linalg.norm(forces, axis=1)))

def kinetic_temperature_from_indices(atoms, indices):
    indices = np.asarray(indices, dtype=int)
    if len(indices) == 0:
        return np.nan
    masses = atoms.get_masses()[indices]
    velocities = atoms.get_velocities()
    if velocities is None:
        return np.nan
    velocities = velocities[indices]
    total_mv2 = np.sum(atoms.get_masses()[:, None] * atoms.get_velocities()**2)
    if total_mv2 <= 0.0:
        return 0.0
    total_ekin = atoms.get_kinetic_energy()
    selected_mv2 = np.sum(masses[:, None] * velocities**2)
    selected_ekin = total_ekin * selected_mv2 / total_mv2
    dof = 3 * len(indices)
    return float(2.0 * selected_ekin / (dof * units.kB))

def zero_fixed_momenta(atoms, fixed_indices):
    momenta = atoms.get_momenta()
    momenta[fixed_indices, :] = 0.0
    atoms.set_momenta(momenta, apply_constraint=False)

def parse_args():
    parser = argparse.ArgumentParser(description="Unified ConstP / ConstQ MD using MACE-short + QEq.")
    parser.add_argument("--mode", choices=["constp", "constq", "full"], default="constp")
    parser.add_argument("--structure", default=str(DEFAULT_STRUCTURE))
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE_K)
    parser.add_argument("--steps", type=int, default=DEFAULT_MD_STEPS)
    parser.add_argument("--timestep-fs", type=float, default=DEFAULT_TIMESTEP_FS)
    parser.add_argument("--ttime-fs", type=float, default=DEFAULT_TTIME_FS)
    parser.add_argument("--save-every", type=int, default=DEFAULT_SAVE_EVERY)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--planes-per-side", type=int, default=CONSTP_FIXED_PLANES_PER_SIDE)
    parser.add_argument("--plane-tol", type=float, default=CONSTP_FIXED_PLANE_TOL)
    parser.add_argument("--qeq-tol", type=float, default=PRODUCTION_QEQ_TOL)
    parser.add_argument("--pme-points-per-ang", type=float, default=PRODUCTION_PME_POINTS_PER_ANG)
    parser.add_argument("--cn-cadence", type=int, default=DEFAULT_CN_CADENCE)
    parser.add_argument("--bottom-external-chi", type=float, default=DEFAULT_BOTTOM_EXTERNAL_CHI)
    parser.add_argument("--upper-external-chi", type=float, default=DEFAULT_UPPER_EXTERNAL_CHI)
    parser.add_argument("--qeq-update-every", type=int, default=DEFAULT_QEQ_UPDATE_EVERY)
    parser.add_argument("--pair-capacity-factor", type=float, default=DEFAULT_PAIR_CAPACITY_FACTOR)
    parser.add_argument("--max-pairs", type=int, default=DEFAULT_MAX_PAIRS)
    parser.add_argument("--active-metal", default=None)
    parser.add_argument("--rcut-metal", type=float, default=None)
    parser.add_argument("--anion-cutoffs", default=None)
    parser.add_argument("--rcut-anion", type=float, default=CONSTP_RCUT_ANION)
    parser.add_argument("--rcut-lili", type=float, default=CONSTP_RCUT_LILI)
    parser.add_argument("--anion-symbol", default=CONSTP_ANION_ELEMENT)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    return parser.parse_args()

def main():
    args = parse_args()
    structure_file = Path(args.structure)
    model_file = Path(args.model)
    output_dir = Path(args.output)

    if not structure_file.exists():
        raise FileNotFoundError(f"Structure not found: {structure_file}")
    if not model_file.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_file}")
    if args.steps <= 0 or args.timestep_fs <= 0.0 or args.ttime_fs <= 0.0:
        raise ValueError("Invalid integration step or time constant.")
    if args.qeq_update_every <= 0:
        raise ValueError("qeq_update_every must be >= 1.")

    output_dir.mkdir(parents=True, exist_ok=True)
    atoms = read(structure_file, index=0)
    natoms = len(atoms)

    active_metal = args.active_metal or getattr(cfg, "ACTIVE_METAL", "Li")
    rcut_metal = args.rcut_metal if args.rcut_metal is not None else (args.rcut_lili if args.rcut_lili is not None else getattr(cfg, "RCUT_METAL", 3.50))
    if args.anion_cutoffs is not None:
        anion_cutoffs = parse_anion_cutoffs(args.anion_cutoffs)
    elif cfg is not None and hasattr(cfg, "ANION_CUTOFFS"):
        anion_cutoffs = {str(k): float(v) for k, v in cfg.ANION_CUTOFFS.items()}
    else:
        anion_cutoffs = {args.anion_symbol: args.rcut_anion}

    if args.mode == "constp" or args.planes_per_side > 0:
        fixed_bottom_indices, fixed_upper_indices, fixed_indices = identify_fixed_metal_planes(
            atoms,
            metal_symbol=active_metal,
            planes_per_side=args.planes_per_side,
            plane_tol=args.plane_tol,
        )
        atoms.set_constraint(FixAtoms(indices=fixed_indices))
        mobile_indices = np.setdiff1d(np.arange(natoms, dtype=int), fixed_indices, assume_unique=False)
    else:
        fixed_bottom_indices = np.array([], dtype=int)
        fixed_upper_indices = np.array([], dtype=int)
        fixed_indices = np.array([], dtype=int)
        mobile_indices = np.arange(natoms, dtype=int)

    if len(mobile_indices) == 0:
        raise RuntimeError("All atoms are fixed; no mobile atoms remain.")

    calc = MACEQEqCalculator(
        model_path=model_file,
        atoms=atoms,
        mode=args.mode,
        fixed_indices=fixed_indices,
        fixed_bottom_indices=fixed_bottom_indices,
        fixed_upper_indices=fixed_upper_indices,
        device=args.device,
        capacity_factor=args.pair_capacity_factor,
        max_pairs=args.max_pairs,
        bottom_external_chi=args.bottom_external_chi,
        upper_external_chi=args.upper_external_chi,
        qeq_update_every=args.qeq_update_every,
        qeq_tol=args.qeq_tol,
        pme_points_per_ang=args.pme_points_per_ang,
        cn_cadence=args.cn_cadence,
        active_metal=active_metal,
        rcut_metal=rcut_metal,
        anion_cutoffs=anion_cutoffs,
        rcut_anion=args.rcut_anion,
        rcut_lili=rcut_metal,
        anion_symbol=args.anion_symbol,
    )
    atoms.calc = calc

    rng = np.random.default_rng(args.seed)
    MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature, rng=rng)
    if len(fixed_indices) > 0:
        zero_fixed_momenta(atoms, fixed_indices)

    initial_energy = float(atoms.get_potential_energy())
    initial_forces = np.asarray(atoms.get_forces(), dtype=np.float64)
    if not np.isfinite(initial_energy) or not np.all(np.isfinite(initial_forces)):
        raise RuntimeError("Non-finite initial state detected.")

    dyn = FixAtomsAwareNPT(
        atoms,
        timestep=args.timestep_fs * units.fs,
        temperature_K=args.temperature,
        ttime=args.ttime_fs * units.fs,
        pfactor=None,
    )

    traj_file = output_dir / "md.traj"
    extxyz_file = output_dir / "md.extxyz"
    thermo_file = output_dir / "thermo.csv"

    for path in [traj_file, extxyz_file, thermo_file]:
        if path.exists():
            path.unlink()

    traj = Trajectory(traj_file, "w")
    thermo_handle = open(thermo_file, "w", newline="")
    thermo = csv.writer(thermo_handle)
    thermo.writerow([
        "step",
        "time_ps",
        "temperature_ASE_K",
        "temperature_mobile_K",
        "Epot_eV",
        "Ekin_eV",
        "Etot_eV",
        "Fmax_eV_A",
        "qeq_energy_eV",
        "qeq_sum_e",
        "qeq_max_abs_q_e",
        "qeq_updated",
        "qeq_eval_counter",
        "qeq_iterations",
        "qeq_solve_s",
        "qeq_force_s",
        "qeq_real_pairs",
        "qeq_pair_capacity",
        "constp_n_electrode",
        "constp_n_bottom",
        "constp_n_upper",
        "constp_n_fixed_in_electrode",
        "constp_n_enter",
        "constp_n_leave",
        "constp_n_switch",
        "constp_calc_evaluations",
        "constp_total_enter",
        "constp_total_leave",
        "constp_total_switch",
        "max_fixed_momentum",
    ])

    def save_frame():
        step = dyn.get_number_of_steps()
        time_ps = step * args.timestep_fs / 1000.0
        epot = float(atoms.get_potential_energy())
        ekin = float(atoms.get_kinetic_energy())
        temperature_ase = float(atoms.get_temperature())
        temperature_mobile = kinetic_temperature_from_indices(atoms, mobile_indices)
        forces = np.asarray(atoms.get_forces(), dtype=np.float64)
        fmax = force_max(forces)

        max_fixed_p = float(np.max(np.linalg.norm(atoms.get_momenta()[fixed_indices], axis=1))) if len(fixed_indices) > 0 else 0.0

        qeq_energy = np.nan
        qeq_sum = np.nan
        qeq_max = np.nan
        qeq_updated = False
        qeq_eval_counter = -1
        qeq_iterations = -1
        qeq_solve_s = np.nan
        qeq_force_s = np.nan
        qeq_pairs = -1
        qeq_capacity = -1

        constp_n_electrode = -1
        constp_n_bottom = -1
        constp_n_upper = -1
        constp_n_fixed_in_electrode = -1
        constp_n_enter = -1
        constp_n_leave = -1
        constp_n_switch = -1
        constp_calc_evaluations = -1
        constp_total_enter = -1
        constp_total_leave = -1
        constp_total_switch = -1

        if calc.last_qeq is not None:
            qeq = calc.last_qeq
            charges = np.asarray(qeq["charges"], dtype=np.float64)
            qeq_energy = float(qeq["energy"])
            qeq_sum = float(charges.sum())
            qeq_max = float(np.max(np.abs(charges)))
            qeq_updated = bool(qeq.get("qeq_updated", True))
            qeq_eval_counter = int(qeq.get("qeq_eval_counter", -1))
            qeq_iterations = int(qeq["iterations"]) if qeq_updated else 0
            qeq_solve_s = float(qeq["solve_s"]) if qeq_updated else 0.0
            qeq_force_s = float(qeq["force_s"]) if qeq_updated else 0.0
            qeq_pairs = int(qeq["n_pairs"])
            qeq_capacity = int(qeq["pair_capacity"])
            atoms.arrays["qeq_charges"] = charges.copy()

        if calc.last_constp_diagnostics is not None:
            constp_diag = calc.last_constp_diagnostics
            electrode_indices = np.asarray(constp_diag["electrode_indices"], dtype=int)
            bottom_indices = np.asarray(constp_diag["bottom_indices"], dtype=int)
            upper_indices = np.asarray(constp_diag["upper_indices"], dtype=int)
            constp_n_electrode = len(electrode_indices)
            constp_n_bottom = len(bottom_indices)
            constp_n_upper = len(upper_indices)
            constp_n_fixed_in_electrode = len(np.intersect1d(fixed_indices, electrode_indices))
            constp_n_enter = int(constp_diag["n_enter"])
            constp_n_leave = int(constp_diag["n_leave"])
            constp_n_switch = int(constp_diag["n_switch"])
            constp_calc_evaluations = int(constp_diag["n_evaluations"])
            constp_total_enter = int(constp_diag["total_enter"])
            constp_total_leave = int(constp_diag["total_leave"])
            constp_total_switch = int(constp_diag["total_switch"])

            if constp_n_fixed_in_electrode != len(fixed_indices):
                raise RuntimeError("Electrode assignment lost fixed seeds.")

        thermo.writerow([
            step,
            time_ps,
            temperature_ase,
            temperature_mobile,
            epot,
            ekin,
            epot + ekin,
            fmax,
            qeq_energy,
            qeq_sum,
            qeq_max,
            int(qeq_updated),
            qeq_eval_counter,
            qeq_iterations,
            qeq_solve_s,
            qeq_force_s,
            qeq_pairs,
            qeq_capacity,
            constp_n_electrode,
            constp_n_bottom,
            constp_n_upper,
            constp_n_fixed_in_electrode,
            constp_n_enter,
            constp_n_leave,
            constp_n_switch,
            constp_calc_evaluations,
            constp_total_enter,
            constp_total_leave,
            constp_total_switch,
            max_fixed_p,
        ])
        thermo_handle.flush()
        traj.write(atoms)
        write(extxyz_file, atoms, format="extxyz", append=True)

        if not np.isfinite(epot) or not np.all(np.isfinite(forces)) or not np.isfinite(temperature_mobile):
            raise RuntimeError(f"Non-finite trajectory step {step}.")
        if calc.last_qeq is not None:
            if abs(qeq_sum) > 1.0e-6:
                raise RuntimeError(f"Neutrality broken at step {step}: sum(q)={qeq_sum}")
            if qeq_pairs >= qeq_capacity:
                raise RuntimeError(f"QEq pair list exhausted at step {step}.")
        if temperature_mobile > 3000.0 or fmax > 100.0:
            raise RuntimeError(f"Instability abort at step {step}: Tmob={temperature_mobile:.1f} K, Fmax={fmax:.3f} eV/A.")

    dyn.attach(save_frame, interval=args.save_every)

    try:
        dyn.run(args.steps)
    finally:
        thermo_handle.close()
        traj.close()

    if calc.electrode_tracker is not None:
        switch_csv = output_dir / "switching_atoms.csv"
        tracker = calc.electrode_tracker
        with open(switch_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["atom_index", "element", "z_initial_A", "enter_count", "leave_count", "total_switch_count"])
            symbols_now = np.asarray(atoms.get_chemical_symbols())
            total_counts = tracker.enter_counts + tracker.leave_counts
            switched_atoms = np.where(total_counts > 0)[0]
            order = switched_atoms[np.argsort(total_counts[switched_atoms])[::-1]]
            for idx in order:
                writer.writerow([
                    int(idx),
                    str(symbols_now[idx]),
                    float(calc.initial_z[idx]),
                    int(tracker.enter_counts[idx]),
                    int(tracker.leave_counts[idx]),
                    int(total_counts[idx]),
                ])

if __name__ == "__main__":
    main()
