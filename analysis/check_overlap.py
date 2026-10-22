#!/usr/bin/env python3

from ase.io import iread
from ase.neighborlist import neighbor_list

FILE = "/work/hoang0000/Li_Li3N/md_validation/2/md.extxyz"
FRAME_DT_PS = 0.01
CHECK_EVERY_PS = 0.1
CUTOFF_A = 1.0

step = max(1, round(CHECK_EVERY_PS / FRAME_DT_PS))
checked = 0
failed = 0
global_min = float("inf")
global_info = None

print(f"File: {FILE}")
print(f"Frame interval: {FRAME_DT_PS} ps")
print(f"Check interval: {step * FRAME_DT_PS:g} ps")
print(f"Overlap threshold: {CUTOFF_A} Ã…")
print("-" * 75)

for frame, atoms in enumerate(iread(FILE, index=":")):
    if frame % step:
        continue

    checked += 1
    time_ps = frame * FRAME_DT_PS

    i, j, d = neighbor_list("ijd", atoms, CUTOFF_A)

    pairs = [(int(a), int(b), float(r))
             for a, b, r in zip(i, j, d) if a < b]

    if not pairs:
        continue

    failed += 1
    symbols = atoms.get_chemical_symbols()

    for a, b, r in sorted(pairs, key=lambda x: x[2]):
        if r < global_min:
            global_min = r
            global_info = (time_ps, frame, a, b, symbols[a], symbols[b])

        print(
            f"t={time_ps:9.3f} ps | frame={frame:6d} | "
            f"{symbols[a]}[{a}]-{symbols[b]}[{b}] | "
            f"d={r:.4f} Ã…"
        )

print("-" * 75)
print(f"Frames checked: {checked}")
print(f"Frames with overlap: {failed}")

if global_info:
    t, frame, a, b, sa, sb = global_info
    print(
        f"Minimum detected: {global_min:.4f} Ã… | "
        f"{sa}[{a}]-{sb}[{b}] | frame={frame} | t={t:.3f} ps"
    )
else:
    print(f"No pairs below {CUTOFF_A} Ã… in checked frames.")
