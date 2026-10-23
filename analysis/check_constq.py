#!/usr/bin/env python3

import numpy as np
from ase.constraints import FixAtoms
from ase.io import iread
from ase.neighborlist import neighbor_list
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

FILE = "/work/hoang0000/Li_Li6PS5Cl/md_validation/constQ/md.extxyz"

TIMESTEP_FS = 1.0
SAVE_EVERY = 100
SAMPLE_EVERY_PS = 0.5

BASELINE_PS = 10.0
WINDOW_PS = 5.0
CONSECUTIVE_WINDOWS = 4
RELATIVE_TOL = 0.10

RCUT_METAL = 3.50
OTHER_CUTOFFS = {"N": 2.50, "P": 2.80, "S": 2.90, "Cl": 3.05}


def get_seeds(atoms):
    fixed = []
    for constraint in atoms.constraints:
        if isinstance(constraint, FixAtoms):
            fixed.extend(constraint.get_indices())

    fixed = np.unique(fixed).astype(int)
    symbols = np.asarray(atoms.get_chemical_symbols())

    if not len(fixed) or not np.all(symbols[fixed] == "Li"):
        raise ValueError("Trajectory must contain fixed Li electrode seeds.")

    z = atoms.positions[:, 2]
    midpoint = 0.5 * (z[fixed].min() + z[fixed].max())
    bottom = fixed[z[fixed] < midpoint]
    upper = fixed[z[fixed] > midpoint]

    if not len(bottom) or not len(upper):
        raise ValueError("Cannot identify both fixed seed layers.")

    return bottom, upper


def count_electrode_li(atoms, bottom, upper):
    symbols = np.asarray(atoms.get_chemical_symbols())
    other = set(symbols) - {"Li"}

    missing = other - set(OTHER_CUTOFFS)
    if missing:
        raise ValueError(f"Missing coordination cutoffs: {sorted(missing)}")

    max_rc = max([RCUT_METAL] + [OTHER_CUTOFFS[s] for s in other])
    i, j, d = neighbor_list("ijd", atoms, max_rc, self_interaction=False)

    is_li = symbols == "Li"
    other_bonds = np.zeros(len(i), dtype=bool)
    for symbol in other:
        other_bonds |= (
            is_li[i]
            & (symbols[j] == symbol)
            & (d <= OTHER_CUTOFFS[symbol])
        )

    cn_other = np.bincount(i[other_bonds], minlength=len(atoms))
    candidate = is_li & (cn_other <= 1)
    candidate[bottom] = True
    candidate[upper] = True

    bonds = candidate[i] & candidate[j] & (d <= RCUT_METAL)
    graph = csr_matrix(
        (np.ones(np.count_nonzero(bonds), dtype=bool), (i[bonds], j[bonds])),
        shape=(len(atoms), len(atoms)),
    )
    _, labels = connected_components(graph, directed=False)

    bottom_labels = np.unique(labels[bottom])
    upper_labels = np.unique(labels[upper])

    if np.intersect1d(bottom_labels, upper_labels).size:
        raise RuntimeError("Candidate graph connects both electrodes.")

    return np.array([
        np.count_nonzero(candidate & np.isin(labels, bottom_labels)),
        np.count_nonzero(candidate & np.isin(labels, upper_labels)),
    ], dtype=float)


def main():
    frame_dt = TIMESTEP_FS * SAVE_EVERY / 1000.0
    baseline_frames = round(BASELINE_PS / frame_dt)
    window_frames = round(WINDOW_PS / frame_dt)

    if (
        baseline_frames < 1
        or window_frames < 1
        or not np.isclose(baseline_frames * frame_dt, BASELINE_PS)
        or not np.isclose(window_frames * frame_dt, WINDOW_PS)
    ):
        raise ValueError("Analysis times must align with saved frames.")

    bottom = upper = None
    initial_numbers = None
    initial_count = previous_count = baseline_rate = None
    consecutive = 0
    first_stop = None
    last_time = None

    print(
        f"{'Window(ps)':>12} {'RateB(Li/ps)':>14} "
        f"{'RateU(Li/ps)':>14} {'Status':>9} {'Streak':>7}"
    )

    for frame, atoms in enumerate(iread(FILE, index=":")):
        selected = (
            frame == 0
            or (
                frame >= baseline_frames
                and (frame - baseline_frames) % window_frames == 0
            )
        )
        if not selected:
            continue

        expected_time = frame * frame_dt
        t = float(atoms.info.get("time_ps", expected_time))
        if not np.isfinite(t) or not np.isclose(
            t, expected_time, rtol=0.0, atol=1e-6
        ):
            raise ValueError("Trajectory time does not match frame indexing.")

        if frame == 0:
            bottom, upper = get_seeds(atoms)
            initial_numbers = atoms.numbers.copy()

        if not np.array_equal(atoms.numbers, initial_numbers):
            raise ValueError("Atom count or ordering changed.")

        current_count = count_electrode_li(atoms, bottom, upper)

        if frame == 0:
            initial_count = current_count
            continue

        if frame == baseline_frames:
            baseline_rate = (current_count - initial_count) / BASELINE_PS
            previous_count = current_count
            last_time = t
            print(
                f"Baseline 0-{BASELINE_PS:g} ps: "
                f"bottom={baseline_rate[0]:.6f}, "
                f"upper={baseline_rate[1]:.6f} Li/ps"
            )
            continue

        rate = (current_count - previous_count) / WINDOW_PS
        stable_sides = (
            (np.abs(rate) <= 1.0)
            | (np.abs(rate) < RELATIVE_TOL * np.abs(baseline_rate))
        )
        stable = bool(np.all(stable_sides))
        consecutive = consecutive + 1 if stable else 0

        print(
            f"{last_time:5.0f}-{t:<6.0f} "
            f"{rate[0]:14.6f} {rate[1]:14.6f} "
            f"{'STABLE' if stable else 'ACTIVE':>9} "
            f"{consecutive:7d}"
        )

        previous_count = current_count
        last_time = t

        if consecutive >= CONSECUTIVE_WINDOWS and first_stop is None:
            first_stop = t

    if baseline_rate is None:
        print("DECISION: INSUFFICIENT DATA")
    else:
        if first_stop is not None:
            print(f"First stop criterion reached: {first_stop:.2f} ps")
        print(
            "DECISION: STOP CONSTQ"
            if consecutive >= CONSECUTIVE_WINDOWS
            else "DECISION: CONTINUE CONSTQ"
        )


if __name__ == "__main__":
    main()
