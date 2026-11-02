#!/usr/bin/env python3

import json
import sys
from pathlib import Path

import numpy as np
from ase.io import iread, write
from ase.neighborlist import neighbor_list

ANION_CUTOFFS = {
    "N": 2.50,
    "P": 2.80,
    "S": 2.90,
    "Cl": 3.05,
}

def main():

    if len(sys.argv) != 2:
        print(f"Usage: python {Path(__file__).name} structure.vasp")
        sys.exit(1)

    input_file = Path(sys.argv[1]).resolve()

    if not input_file.exists():
        raise FileNotFoundError(input_file)

    script_dir = Path(__file__).resolve().parent

    output_file = script_dir / f"{input_file.stem}_Li_metal.vasp"

    metadata_file = input_file.parent / "md_run_metadata.json"

    frames = iread(input_file, index=":")
    atoms = next(frames, None)

    if atoms is None:
        raise ValueError(f"No frame found in {input_file}")

    if next(frames, None) is not None:
        raise ValueError(
            f"Input must contain exactly one frame: {input_file}"
        )

    metadata = {}

    if metadata_file.exists():
        metadata = json.loads(metadata_file.read_text())

    cutoffs = metadata.get("anion_cutoffs")

    if cutoffs is None:
        cutoffs = ANION_CUTOFFS
    elif isinstance(cutoffs, str):
        cutoffs = json.loads(cutoffs)

    cutoffs = {
        str(key): float(value)
        for key, value in cutoffs.items()
    }

    symbols = np.asarray(atoms.get_chemical_symbols())
    other_elements = set(symbols) - {"Li"}

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

    for key in (
        "fixed_bottom_indices",
        "fixed_upper_indices",
    ):
        indices = np.asarray(
            metadata.get(key, []),
            dtype=int,
        )

        keep[indices] = True

    selected = atoms[keep]

    write(
        output_file,
        selected,
        format="vasp",
        direct=True,
        sort=False,
    )

    print(f"Output: {output_file}")

if __name__ == "__main__":
    main()
