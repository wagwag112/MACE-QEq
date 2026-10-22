import sys
import numpy as np
from ase.io import read
from ase.neighborlist import neighbor_list

traj_file = sys.argv[1] if len(sys.argv) > 1 else "md.traj"
traj = read(traj_file, index=":")

atoms_init = traj[0]
atoms_curr = traj[-1]
symbols = np.array(atoms_curr.get_chemical_symbols())

box = atoms_curr.get_cell()
z_mid = box[2, 2] / 2.0

li_idx = np.where(symbols == "Li")[0]
li_set = set(li_idx)

ANION_CUTOFFS = {
    "N": 2.50,
    "P": 2.80,
    "S": 2.90,
    "Cl": 3.05,
}

ELECTROLYTE_CN_MIN = 2

def get_bulk_li(atoms):
    i_li, j_li, d_li = neighbor_list(
        "ijd", atoms, max(ANION_CUTOFFS.values())
    )
    symbols_local = np.array(atoms.get_chemical_symbols())
    cn = {idx: 0 for idx in li_idx}

    for i, j, d in zip(i_li, j_li, d_li):
        if i in li_set and symbols_local[j] in ANION_CUTOFFS:
            if d <= ANION_CUTOFFS[symbols_local[j]]:
                cn[i] += 1

    return cn

cn_init = get_bulk_li(atoms_init)
cn_curr = get_bulk_li(atoms_curr)

pos_init = atoms_init.positions
pos_curr = atoms_curr.positions

# Classify electrodes along Z
# Bottom Li: z < z_mid | Upper Li: z > z_mid
# Li with CN_anion < 2 is classified as electrode Li
bottom_li_init = [
    i for i in li_idx
    if pos_init[i, 2] < z_mid and cn_init[i] < ELECTROLYTE_CN_MIN
]
bottom_li_curr = [
    i for i in li_idx
    if pos_curr[i, 2] < z_mid and cn_curr[i] < ELECTROLYTE_CN_MIN
]
delta_bottom = len(bottom_li_curr) - len(bottom_li_init)

upper_li_init = [
    i for i in li_idx
    if pos_init[i, 2] > z_mid and cn_init[i] < ELECTROLYTE_CN_MIN
]
upper_li_curr = [
    i for i in li_idx
    if pos_curr[i, 2] > z_mid and cn_curr[i] < ELECTROLYTE_CN_MIN
]
delta_upper = len(upper_li_curr) - len(upper_li_init)

# Bulk Li thickness along Z
if len(bottom_li_curr) > 0:
    thick_bottom = (
        np.max(pos_curr[bottom_li_curr, 2])
        - np.min(pos_curr[bottom_li_curr, 2])
    )
else:
    thick_bottom = 0.0

if len(upper_li_curr) > 0:
    thick_upper = (
        np.max(pos_curr[upper_li_curr, 2])
        - np.min(pos_curr[upper_li_curr, 2])
    )
else:
    thick_upper = 0.0

# LPSC core integrity along Z
p_core = [
    i for i, s in enumerate(symbols)
    if s == "P" and abs(pos_curr[i, 2] - z_mid) < 8.0
]

s_indices = set(np.where(symbols == "S")[0])
i_p, j_p, d_p = neighbor_list("ijd", atoms_curr, 2.45)

cn_p = {idx: 0 for idx in p_core}
for i, j in zip(i_p, j_p):
    if i in cn_p and j in s_indices:
        cn_p[i] += 1

mean_cn_p = (
    np.mean(list(cn_p.values()))
    if len(p_core) > 0 else 0.0
)

print(f"=== ELECTRODE ANALYSIS ALONG Z (Frames: {len(traj)}) ===")
print(f"Electrolyte-like Li criterion: CN_anion >= {ELECTROLYTE_CN_MIN}")

print(f"\nBottom Electrode Li (z < {z_mid:.1f} A):")
print(
    f"  - Electrode atoms (CN_anion < {ELECTROLYTE_CN_MIN}): "
    f"Start = {len(bottom_li_init)}, End = {len(bottom_li_curr)}"
)
print(f"  - Atom count change (Delta N): {delta_bottom:+d} atoms")
print(f"  - Electrode thickness: {thick_bottom:.2f} A")

print(f"\nUpper Electrode Li (z > {z_mid:.1f} A):")
print(
    f"  - Electrode atoms (CN_anion < {ELECTROLYTE_CN_MIN}): "
    f"Start = {len(upper_li_init)}, End = {len(upper_li_curr)}"
)
print(f"  - Atom count change (Delta N): {delta_upper:+d} atoms")
print(f"  - Electrode thickness: {thick_upper:.2f} A")

print(f"\nLPSC Core Integrity (z_mid +/- 8 A):")
print(f"  - P atoms in core: {len(p_core)}")
print(f"  - Mean P-S coordination: {mean_cn_p:.2f}")
