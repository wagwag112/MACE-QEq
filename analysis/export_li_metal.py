#!/usr/bin/env python3

import json
from pathlib import Path

import numpy as np
from ase.io import iread, write
from ase.neighborlist import neighbor_list


INPUT_FILE = Path("/work/hoang0000/Li_Li3N/md_validation/6/md_frame41.vasp")
OUTPUT_FILE = Path("/work/hoang0000/Li_Li3N/md_validation/6/Li_metal.vasp")
METADATA_FILE = Path("/work/hoang0000/Li_Li3N/md_validation/6/md_run_metadata.json")

ANION_CUTOFFS = {
    "N": 2.50,
    "P": 2.80,
    "S": 2.90,
    "Cl": 3.05,
}


def main():
    if INPUT_FILE.resolve() == OUTPUT_FILE.resolve():
        raise ValueError("Input and output files must be different.")

    frames = iread(INPUT_FILE, index=":")
    atoms = next(frames, None)

    if atoms is None:
        raise ValueError(f"No frame found in {INPUT_FILE}")

    if next(frames, None) is not None:
        raise ValueError(f"Input must contain exactly one frame: {INPUT_FILE}")

    metadata = {}
    if METADATA_FILE is not None:
        metadata = json.loads(METADATA_FILE.read_text())

    cutoffs = metadata.get("anion_cutoffs")
    if cutoffs is None:
        cutoffs = ANION_CUTOFFS
    elif isinstance(cutoffs, str):
        cutoffs = json.loads(cutoffs)

    cutoffs = {str(key): float(value) for key, value in cutoffs.items()}

    if not cutoffs or any(
        not np.isfinite(value) or value <= 0.0
        for value in cutoffs.values()
    ):
        raise ValueError("Anion cutoffs must be finite and positive.")

    symbols = np.asarray(atoms.get_chemical_symbols())
    other_elements = set(symbols) - {"Li"}
    missing_cutoffs = other_elements - set(cutoffs)

    if missing_cutoffs:
        raise ValueError(
            f"Missing coordination cutoffs: {sorted(missing_cutoffs)}"
        )

    pair_i, pair_j, distances = neighbor_list(
        "ijd",
        atoms,
        max(cutoffs.values()),
        self_interaction=False,
    )

    bonded = np.zeros(len(pair_i), dtype=bool)

    for symbol in other_elements:
        bonded |= (
            (symbols[pair_i] == "Li")
            & (symbols[pair_j] == symbol)
            & (distances <= cutoffs[symbol])
        )

    coordination = np.bincount(
        pair_i[bonded],
        minlength=len(atoms),
    )

    keep = (symbols == "Li") & (coordination <= 1)

    for key in ("fixed_bottom_indices", "fixed_upper_indices"):
        indices = np.asarray(metadata.get(key, []), dtype=int)

        if np.any(indices < 0) or np.any(indices >= len(atoms)):
            raise ValueError(f"Invalid {key} in {METADATA_FILE}")

        if np.any(symbols[indices] != "Li"):
            raise ValueError(f"Non-Li atoms in {key}: {METADATA_FILE}")

        keep[indices] = True

    selected = atoms[keep]

    if len(selected) == 0:
        raise RuntimeError("No metallic Li detected.")

    selected.set_constraint()
    write(OUTPUT_FILE, selected, format="vasp", direct=True, sort=False)

    print(f"Input: {INPUT_FILE}")
    print(f"Cutoffs (angstrom): {cutoffs}")
    print(f"Selected Li: {len(selected)} / {np.count_nonzero(symbols == 'Li')}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
