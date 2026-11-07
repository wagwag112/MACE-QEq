import sys
import numpy as np
from ase.io import read
from ase.neighborlist import neighbor_list

traj_file = sys.argv[1] if len(sys.argv) > 1 else "md.traj"
time_ps = float(sys.argv[2]) if len(sys.argv) > 2 else -1
frames_per_ps = 10
traj = read(traj_file, index=":")

if time_ps < 0:
    end_frame = len(traj) - 1
else:
    end_frame = int(round(time_ps * frames_per_ps))

if end_frame >= len(traj) or end_frame < frames_per_ps - 1:
    raise ValueError(f"Invalid time: {time_ps} ps (available frames: {len(traj)})")

atoms_init = traj[0]
frames_curr = traj[end_frame - frames_per_ps + 1:end_frame + 1]
symbols = np.array(atoms_init.get_chemical_symbols())

ANION_CUTOFFS = {
    "N": 2.50,
    "P": 2.80,
    "S": 2.90,
    "Cl": 3.05,
}

ELECTROLYTE_CN_MIN = 2

li_idx = np.where(symbols == "Li")[0]
li_set = set(li_idx)

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

def analyze_frame(atoms):
    cn = get_bulk_li(atoms)
    pos = atoms.positions
    z_mid = atoms.get_cell()[2, 2] / 2.0

    bottom_li = [
        i for i in li_idx
        if pos[i, 2] < z_mid and cn[i] < ELECTROLYTE_CN_MIN
    ]
    upper_li = [
        i for i in li_idx
        if pos[i, 2] > z_mid and cn[i] < ELECTROLYTE_CN_MIN
    ]

    thick_bottom = (
        np.max(pos[bottom_li, 2]) - np.min(pos[bottom_li, 2])
        if len(bottom_li) > 0 else 0.0
    )
    thick_upper = (
        np.max(pos[upper_li, 2]) - np.min(pos[upper_li, 2])
        if len(upper_li) > 0 else 0.0
    )

    return len(bottom_li), len(upper_li), thick_bottom, thick_upper

bottom_init, upper_init, _, _ = analyze_frame(atoms_init)

results = np.array([analyze_frame(atoms) for atoms in frames_curr])
bottom_mean, upper_mean, thick_bottom, thick_upper = np.mean(results, axis=0)

delta_bottom = bottom_mean - bottom_init
delta_upper = upper_mean - upper_init

end_time_ps = end_frame / frames_per_ps
start_time_ps = (end_frame - frames_per_ps + 1) / frames_per_ps

print(f"=== ELECTRODE ANALYSIS ALONG Z (Time: {end_time_ps:.1f} ps) ===")
print(f"Averaging: {len(frames_curr)} frames ({start_time_ps:.1f}-{end_time_ps:.1f} ps)")
print(f"Electrolyte-like Li criterion: CN_anion >= {ELECTROLYTE_CN_MIN}")

print("\nBottom Electrode Li:")
print(
    f"  - Electrode atoms (CN_anion < {ELECTROLYTE_CN_MIN}): "
    f"Start = {bottom_init}, Mean = {bottom_mean:.2f}"
)
print(f"  - Atom count change (Delta N): {delta_bottom:+.2f} atoms")
print(f"  - Mean electrode thickness: {thick_bottom:.2f} A")

print("\nUpper Electrode Li:")
print(
    f"  - Electrode atoms (CN_anion < {ELECTROLYTE_CN_MIN}): "
    f"Start = {upper_init}, Mean = {upper_mean:.2f}"
)
print(f"  - Atom count change (Delta N): {delta_upper:+.2f} atoms")
print(f"  - Mean electrode thickness: {thick_upper:.2f} A")
