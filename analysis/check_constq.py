#!/usr/bin/env python3

import numpy as np
from ase.io import iread

# ===================== PARAMETERS =====================

FILE = "/work/hoang0000/Li_Li6PS5Cl/md_validation/4vs2/md.extxyz"

TIMESTEP_FS = 1.0
SAVE_EVERY = 100

SAMPLE_EVERY_PS = 0.5
WINDOW_PS = 5.0
MIN_TIME_PS = 20.0
CONSECUTIVE_WINDOWS = 3

RELATIVE_TOL = 0.10
MIX_AUTO_STABLE_A = 1.0

INTERFACE_WIDTH_A = 8.0
PENETRATION_MARGIN_A = 1.0

# ======================================================


def detect_geometry(atoms):
    symbols = np.array(atoms.get_chemical_symbols())
    nonli = symbols != "Li"
    li = ~nonli

    if not np.any(nonli):
        raise ValueError("No non-Li atoms found.")

    candidates = []

    for axis in range(3):
        x = atoms.positions[:, axis]

        lo = np.min(x[nonli])
        hi = np.max(x[nonli])

        left = li & (x < lo)
        right = li & (x > hi)

        if np.any(left) and np.any(right):
            candidates.append((
                min(np.sum(left), np.sum(right)),
                axis, lo, hi, left, right
            ))

    if not candidates:
        raise ValueError(
            "Cannot detect Li|electrolyte|Li geometry."
        )

    _, axis, lo, hi, left, right = max(
        candidates, key=lambda x: x[0]
    )

    electrolyte_li = li & ~left & ~right

    return {
        "axis": axis,
        "lo": lo,
        "hi": hi,
        "left_li": left,
        "right_li": right,
        "electrolyte_li": electrolyte_li,
        "nonli": nonli,
    }


def unwrap_positions(previous, current, cell_length):
    delta = current - previous
    delta -= cell_length * np.round(delta / cell_length)
    return delta


def measure(x, geometry):
    lo = geometry["lo"]
    hi = geometry["hi"]

    left_li = geometry["left_li"]
    right_li = geometry["right_li"]
    nonli = geometry["nonli"]

    margin = PENETRATION_MARGIN_A
    width = INTERFACE_WIDTH_A

    result = {}

    # Left interface
    left_nonli_pen = np.sum(
        nonli & (x < lo - margin)
    )

    left_li_pen = np.sum(
        left_li & (x > lo + margin)
    )

    left_nonli_region = x[
        nonli & (x >= lo - width) & (x <= lo + width)
    ]

    left_li_region = x[
        left_li & (x >= lo - width) & (x <= lo + width)
    ]

    # Right interface
    right_nonli_pen = np.sum(
        nonli & (x > hi + margin)
    )

    right_li_pen = np.sum(
        right_li & (x < hi - margin)
    )

    right_nonli_region = x[
        nonli & (x >= hi - width) & (x <= hi + width)
    ]

    right_li_region = x[
        right_li & (x >= hi - width) & (x <= hi + width)
    ]

    def mixing_width(a, b):
        if len(a) < 2 or len(b) < 2:
            return 0.0

        a_low, a_high = np.percentile(a, [10, 90])
        b_low, b_high = np.percentile(b, [10, 90])

        return max(
            0.0,
            min(a_high, b_high) - max(a_low, b_low)
        )

    result["left"] = np.array([
        left_nonli_pen,
        left_li_pen,
        mixing_width(left_nonli_region, left_li_region)
    ], dtype=float)

    result["right"] = np.array([
        right_nonli_pen,
        right_li_pen,
        mixing_width(right_nonli_region, right_li_region)
    ], dtype=float)

    return result


def stable(previous, current):
    if current[2] < MIX_AUTO_STABLE_A:
        return True

    for idx in range(3):
        old = previous[idx]
        new = current[idx]

        if old == 0:
            if new != 0:
                return False
            continue

        relative_change = abs(new - old) / abs(old)

        if relative_change > RELATIVE_TOL:
            return False

    return True


def main():
    frame_dt = TIMESTEP_FS * SAVE_EVERY / 1000.0

    stride = max(
        1, round(SAMPLE_EVERY_PS / frame_dt)
    )

    sample_dt = stride * frame_dt

    if sample_dt > WINDOW_PS:
        raise ValueError(
            "Sampling interval exceeds analysis window."
        )

    trajectory = iread(FILE, index=f"::{stride}")

    first = next(trajectory, None)

    if first is None:
        raise ValueError("Empty trajectory.")

    geometry = detect_geometry(first)

    axis = geometry["axis"]
    length = np.linalg.norm(first.cell[axis])

    if length <= 0:
        raise ValueError("Invalid cell length.")

    previous_wrapped = first.positions[:, axis].copy()
    unwrapped = previous_wrapped.copy()

    records = []

    records.append((
        0.0,
        measure(unwrapped, geometry)
    ))

    for k, atoms in enumerate(trajectory, start=1):
        if len(atoms) != len(first):
            raise ValueError("Atom count changed.")

        current = atoms.positions[:, axis]

        displacement = unwrap_positions(
            previous_wrapped, current, length
        )

        unwrapped += displacement
        previous_wrapped = current.copy()

        t = k * sample_dt

        records.append((
            t,
            measure(unwrapped, geometry)
        ))

    print("=" * 92)
    print("CONSTQ INTERFACE EQUILIBRATION ANALYSIS")
    print("=" * 92)
    print(f"File             : {FILE}")
    print(f"Detected axis    : {'xyz'[axis]}")
    print(f"Electrolyte range: {geometry['lo']:.2f} - {geometry['hi']:.2f} Ã…")
    print(f"Frame dt         : {frame_dt:.4f} ps")
    print(f"Sampling interval: {sample_dt:.4f} ps")
    print(f"Frames analyzed  : {len(records)}")
    print(f"Duration         : {records[-1][0]:.2f} ps")
    print("-" * 92)

    print(
        f"{'Window':>12} "
        f"{'Side':>6} "
        f"{'NonLi->Li':>12} "
        f"{'Li->SE':>12} "
        f"{'Mix(A)':>10} "
        f"{'Status':>12}"
    )

    window_results = []

    total_time = records[-1][0]
    n_windows = int((total_time + 1e-8) // WINDOW_PS)

    for k in range(n_windows):
        start = k * WINDOW_PS
        end = (k + 1) * WINDOW_PS

        selected = [
            values for t, values in records
            if start < t <= end + 1e-8
        ]

        if not selected:
            continue

        means = {
            side: np.mean(
                [v[side] for v in selected], axis=0
            )
            for side in ("left", "right")
        }

        window_results.append(
            (start, end, means)
        )

    consecutive = 0
    first_stop = None

    for k, (start, end, means) in enumerate(window_results):
        eligible = (
            end >= MIN_TIME_PS
            and k > 0
        )

        if eligible:
            previous = window_results[k - 1][2]

            left_ok = stable(
                previous["left"], means["left"]
            )

            right_ok = stable(
                previous["right"], means["right"]
            )

            both_ok = left_ok and right_ok

            consecutive = (
                consecutive + 1 if both_ok else 0
            )
        else:
            left_ok = False
            right_ok = False
            consecutive = 0

        for side, ok in (
            ("left", left_ok),
            ("right", right_ok),
        ):
            values = means[side]

            if not eligible:
                status = "WAIT"
            elif ok:
                status = "STABLE"
            else:
                status = "ACTIVE"

            print(
                f"{start:5.0f}-{end:<5.0f} "
                f"{side:>6} "
                f"{values[0]:12.2f} "
                f"{values[1]:12.2f} "
                f"{values[2]:10.3f} "
                f"{status:>12}"
            )

        if (
            consecutive >= CONSECUTIVE_WINDOWS
            and first_stop is None
        ):
            first_stop = end

    print("=" * 92)

    if first_stop is not None:
        print(
            f"First equilibrium criterion reached: "
            f"{first_stop:.2f} ps"
        )

    if total_time < MIN_TIME_PS:
        print("DECISION: INSUFFICIENT DATA")
    elif consecutive >= CONSECUTIVE_WINDOWS:
        print("DECISION: STOP CONSTQ")
    else:
        print("DECISION: CONTINUE CONSTQ")

    print("=" * 92)


if __name__ == "__main__":
    main()
