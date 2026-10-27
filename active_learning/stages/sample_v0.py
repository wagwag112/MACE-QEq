from __future__ import annotations
import csv
import gc
import os
import sys
from pathlib import Path
import numpy as np
import torch
import multiprocessing as mp
from ase import units
from ase.io import read, write
from ase.geometry import find_mic
from ase.io.trajectory import Trajectory
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from mace.calculators import MACECalculator

SCRIPT_DIR = Path(__file__).resolve().parent
AL_DIR = SCRIPT_DIR.parent
if str(AL_DIR) not in sys.path:
    sys.path.insert(0, str(AL_DIR))

import al_config as cfg

def run_md_simulation(atoms, model_path, output_traj, total_ps, timestep_fs=1.0, temp_k=300.0, save_every=10, seed=20261004, device="cuda:0"):
    total_steps = int(round(total_ps * 1000.0 / timestep_fs))
    expected_frames = total_steps // save_every + 1

    if output_traj.is_file():
        try:
            with Trajectory(output_traj, "r") as t:
                if len(t) >= expected_frames:
                    print(f"[{device}] Trajectory valid on disk: {output_traj.name} ({len(t)}/{expected_frames} frames). Skipping MD.")
                    return
        except Exception:
            pass

    output_traj.parent.mkdir(parents=True, exist_ok=True)
    work = atoms.copy()
    calc = MACECalculator(model_paths=[str(model_path)], device=device, default_dtype="float32")
    work.calc = calc

    rng = np.random.default_rng(seed)
    MaxwellBoltzmannDistribution(work, temperature_K=temp_k, rng=rng)

    dyn = Langevin(work, timestep=timestep_fs * units.fs, temperature_K=temp_k, friction=0.01 / units.fs, rng=rng)

    traj = Trajectory(output_traj, "w")
    work.info["time_ps"] = 0.0
    traj.write(work)

    def record_step():
        step = dyn.get_number_of_steps()
        work.info["time_ps"] = float(step * timestep_fs / 1000.0)
        traj.write(work)

    dyn.attach(record_step, interval=save_every)
    dyn.run(total_steps)
    traj.close()

    del calc, dyn, work
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def extract_sampled_frames(traj_file, count, dt_ps, t_equil_ps, save_every=10, timestep_fs=1.0):
    ps_per_frame = (save_every * timestep_fs) / 1000.0
    start_frame = int(round(t_equil_ps / ps_per_frame))
    step_frames = int(round(dt_ps / ps_per_frame))
    indices = [start_frame + (i + 1) * step_frames for i in range(count)]
    with Trajectory(traj_file, "r") as t:
        if indices[-1] >= len(t):
            raise RuntimeError(f"Index {indices[-1]} exceeds trajectory length {len(t)} in {traj_file}")
        extracted = []
        for idx in indices:
            fr = t[idx]
            fr.calc = None
            extracted.append(fr)
        return extracted

def get_task_specs():
    tasks = []
    for sys_name, spec in sorted(
        cfg.V0_SPECS.items(),
        key=lambda item: item[1]["method"] != "md",
    ):
        for role in cfg.V0_ROLES:
            n = spec["counts"][role]
            is_md = spec["method"] == "md"
            tasks.append({
                "task_id": len(tasks), "system": sys_name, "role": role,
                "method": spec["method"], "struct": spec["struct"][role],
                "n_frames": n,
                "seed": cfg.V0_SEED + 10000 * len(tasks),
                "total_ps": cfg.V0_MD_EQUIL_PS + n * cfg.V0_MD_DT_PS if is_md else 0.0,
                "traj_file": cfg.V0_TRAJ_DIR / f"{sys_name}_{role}.traj",
            })
    return tasks

def generate_bulk_frames(atoms, count, seed):
    rng = np.random.default_rng(seed)
    methods = cfg.V0_BULK_METHODS
    frames = []
    for i in range(count):
        method = methods[i % len(methods)]
        frame = atoms.copy()
        frame.info["sampling_method"] = method
        if method == "strain":
            scales = cfg.V0_STRAIN_SCALE
            j = (i // len(methods)) % (len(scales) - 1)
            scale = float(rng.uniform(scales[j], scales[j + 1]))
            frame.set_cell(atoms.cell * scale, scale_atoms=True)
            frame.info["strain_scale"] = scale
        elif method == "rattle":
            stdev = float(cfg.V0_RATTLE_STDEV_A[(i // len(methods)) % len(cfg.V0_RATTLE_STDEV_A)])
            frame.rattle(stdev=stdev, rng=rng)
            frame.info["rattle_stdev_A"] = stdev
        else:
            amplitude = float(cfg.V0_DISPLACEMENT_A[(i // len(methods)) % len(cfg.V0_DISPLACEMENT_A)])
            li_indices = np.flatnonzero(frame.numbers == 3)
            if not len(li_indices):
                raise RuntimeError("Bulk displacement requires Li atoms")
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            index = int(rng.choice(li_indices))
            frame.positions[index] += amplitude * direction
            frame.info["displacement_A"] = amplitude
        frame.info["sampling_seed"] = seed
        frames.append(frame)
    return frames

def task_output_complete(task):
    role = task["role"]
    system = task["system"]
    count = task["n_frames"]
    manifest = cfg.V0_BOOTSTRAP_DIR / f"v0_manifest_task_{task['task_id']}.csv"
    sampled = cfg.V0_SAMPLED_DIR / f"{role}_sampled_{system}.extxyz"
    if not manifest.is_file() or not sampled.is_file():
        return False
    try:
        with open(manifest, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if len(rows) != count:
            return False
        frames = read(sampled, index=":")
        if len(frames) != count:
            return False
        cal_root = cfg.ROOT / "DFT" / "v0" / "cal"
        for idx, (row, frame) in enumerate(zip(rows, frames)):
            source = f"{role}_{system}_{idx:03d}"
            poscar = cal_root / system / source / "POSCAR"
            if row.get("source") != source or row.get("system") != system or row.get("role") != role:
                return False
            if frame.info.get("source") != source or not poscar.is_file() or poscar.stat().st_size == 0:
                return False
            saved = read(poscar, format="vasp")
            if len(saved) != len(frame):
                return False
            if not np.array_equal(saved.numbers, frame.numbers):
                return False
            if not np.array_equal(saved.pbc, frame.pbc):
                return False
            if not np.allclose(saved.cell.array, frame.cell.array, rtol=0.0, atol=1e-6):
                return False
            delta = saved.positions - frame.positions
            delta -= np.rint(delta @ np.linalg.inv(frame.cell.array)) @ frame.cell.array
            if np.max(np.linalg.norm(delta, axis=1)) > 1e-4:
                return False
        return True
    except (OSError, ValueError, KeyError, IndexError):
        return False


def process_single_task(task, device):
    if task_output_complete(task):
        print(f"SKIP Task {task['task_id']}: validated outputs already exist", flush=True)
        return
    struct_file = task["struct"]
    if not struct_file.is_file():
        raise FileNotFoundError(f"Structure missing: {struct_file}")
    atoms = read(struct_file)
    role, sys_name = task["role"], task["system"]
    print(f"[{device}] START Task {task['task_id']}: {role} {sys_name}", flush=True)
    if task["method"] == "md":
        run_md_simulation(
            atoms, cfg.PRETRAINED_MACE_MODEL, task["traj_file"], task["total_ps"],
            timestep_fs=cfg.V0_MD_TIMESTEP_FS,
            temp_k=cfg.TEMPERATURE_K, save_every=cfg.V0_MD_SAVE_EVERY,
            seed=task["seed"], device=device,
        )
        selected = extract_sampled_frames(
            task["traj_file"], task["n_frames"], cfg.V0_MD_DT_PS,
            cfg.V0_MD_EQUIL_PS, cfg.V0_MD_SAVE_EVERY, cfg.V0_MD_TIMESTEP_FS,
        )
    else:
        selected = generate_bulk_frames(atoms, task["n_frames"], task["seed"])
    cal_root = cfg.ROOT / "DFT" / "v0" / "cal"
    cal_root.mkdir(parents=True, exist_ok=True)
    cfg.V0_SAMPLED_DIR.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for idx, frame in enumerate(selected):
        source = f"{role}_{sys_name}_{idx:03d}"
        frame.calc = None
        frame.info.update(source=source, system=sys_name, al_role=role, al_round=0)
        job_dir = cal_root / sys_name / source
        job_dir.mkdir(parents=True, exist_ok=True)
        write(job_dir / "POSCAR", frame, format="vasp", direct=True, vasp5=True, sort=False)
        manifest_rows.append({
            "round": 0, "parent_round": -1, "system": sys_name,
            "role": role, "source": source, "filename": f"{source}.vasp",
            "time_ps": frame.info.get("time_ps", 0.0),
        })
    write(cfg.V0_SAMPLED_DIR / f"{role}_sampled_{sys_name}.extxyz", selected, format="extxyz")
    slice_manifest = cfg.V0_BOOTSTRAP_DIR / f"v0_manifest_task_{task['task_id']}.csv"
    slice_manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(slice_manifest, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["round", "parent_round", "system", "role", "source", "filename", "time_ps"])
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"[{device}] TASK {task['task_id']} COMPLETE", flush=True)

def finalize_bootstrap(tasks):
    if len(tasks) != 9 or sorted(task["task_id"] for task in tasks) != list(range(9)):
        raise RuntimeError("V0 finalize requires exactly tasks 0-8")

    incomplete = [
        task["task_id"]
        for task in tasks
        if not task_output_complete(task)
    ]
    if incomplete:
        raise RuntimeError(f"V0 finalize blocked: invalid tasks {incomplete}")

    cal_root = cfg.ROOT / "DFT" / "v0" / "cal"
    manifest_file = cfg.V0_BOOTSTRAP_DIR / "v0_selection_manifest.csv"
    fields = ["round", "parent_round", "system", "role", "source", "filename", "time_ps"]
    all_rows = []
    frames_by_role = {role: [] for role in cfg.V0_ROLES}
    expected_by_role = {role: 0 for role in cfg.V0_ROLES}
    seen_sources = set()

    # Validate exact task outputs before writing merged files.
    for task in tasks:
        role = task["role"]
        system = task["system"]
        count = task["n_frames"]
        expected_by_role[role] += count

        slice_file = cfg.V0_BOOTSTRAP_DIR / f"v0_manifest_task_{task['task_id']}.csv"
        sampled_file = cfg.V0_SAMPLED_DIR / f"{role}_sampled_{system}.extxyz"

        if not slice_file.is_file() or not sampled_file.is_file():
            raise RuntimeError(f"Missing sampling output for task {task['task_id']}")

        with open(slice_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != fields:
                raise RuntimeError(f"Invalid manifest columns: {slice_file}")
            rows = list(reader)

        if len(rows) != count:
            raise RuntimeError(f"Task {task['task_id']} manifest count: {len(rows)} != {count}")

        frames = read(sampled_file, index=":")
        if len(frames) != count:
            raise RuntimeError(f"Task {task['task_id']} frame count: {len(frames)} != {count}")

        for index, (row, frame) in enumerate(zip(rows, frames)):
            source = f"{role}_{system}_{index:03d}"
            poscar = cal_root / system / source / "POSCAR"

            if row["source"] != source or row["system"] != system or row["role"] != role:
                raise RuntimeError(f"Manifest mismatch: {source}")
            if row["round"] != "0" or row["parent_round"] != "-1":
                raise RuntimeError(f"Invalid round metadata: {source}")
            if row["filename"] != f"{source}.vasp":
                raise RuntimeError(f"Invalid filename: {source}")
            if frame.info.get("source") != source:
                raise RuntimeError(f"Frame metadata mismatch: {source}")
            if frame.info.get("system") != system or frame.info.get("al_role") != role:
                raise RuntimeError(f"Frame role/system mismatch: {source}")
            if source in seen_sources:
                raise RuntimeError(f"Duplicate source: {source}")
            if not poscar.is_file() or poscar.stat().st_size == 0:
                raise RuntimeError(f"Missing POSCAR: {poscar}")

            poscar_atoms = read(poscar)
            if not np.array_equal(poscar_atoms.numbers, frame.numbers):
                raise RuntimeError(f"POSCAR species mismatch: {source}")
            if not np.array_equal(poscar_atoms.pbc, frame.pbc):
                raise RuntimeError(f"POSCAR PBC mismatch: {source}")
            if np.max(np.abs(poscar_atoms.cell.array - frame.cell.array)) > 1e-8:
                raise RuntimeError(f"POSCAR cell mismatch: {source}")

            delta = poscar_atoms.positions - frame.positions
            delta = find_mic(delta, frame.cell, frame.pbc)[0]
            if np.max(np.linalg.norm(delta, axis=1)) > 1e-8:
                raise RuntimeError(f"POSCAR positions mismatch: {source}")

            seen_sources.add(source)

        all_rows.extend(rows)
        frames_by_role[role].extend(frames)

    if len(all_rows) != cfg.V0_TARGET_COUNT:
        raise RuntimeError(f"Total count: {len(all_rows)} != {cfg.V0_TARGET_COUNT}")

    for role in cfg.V0_ROLES:
        expected = sum(spec["counts"][role] for spec in cfg.V0_SPECS.values())
        if expected_by_role[role] != expected or len(frames_by_role[role]) != expected:
            raise RuntimeError(f"Role count mismatch: {role}")

    # Keep task outputs for reproducibility and debugging.
    cfg.V0_BOOTSTRAP_DIR.mkdir(parents=True, exist_ok=True)
    with open(manifest_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)

    for role in cfg.V0_ROLES:
        write(
            cfg.V0_SAMPLED_DIR / f"{role}_sampled.extxyz",
            frames_by_role[role],
            format="extxyz",
        )

    print(f"SAMPLE V0 COMPLETE: frames={len(all_rows)} manifest={manifest_file}")
    return True

def _mp_worker(task_queue, gpu_id):
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"
    while True:
        try:
            task = task_queue.get(timeout=3)
        except Exception:
            break
        process_single_task(task, device)

def main():
    cfg.validate_config()
    cfg.V0_TRAJ_DIR.mkdir(parents=True, exist_ok=True)
    cfg.V0_SAMPLED_DIR.mkdir(parents=True, exist_ok=True)
    cal_root = cfg.ROOT / "DFT" / "v0" / "cal"
    cal_root.mkdir(parents=True, exist_ok=True)

    tasks = sorted(get_task_specs(), key=lambda x: x["total_ps"], reverse=True)

    # Array workers write disjoint outputs; finalize in a separate invocation.
    if "--finalize" in sys.argv:
        if not finalize_bootstrap(tasks):
            raise RuntimeError("Incomplete sampling tasks")
        return

    if "SLURM_ARRAY_TASK_ID" in os.environ:
        task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
        process_single_task(tasks[task_id], "cuda:0" if tasks[task_id]["method"] == "md" else "cpu")
        return

    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"V0 Bootstrap Manager: {len(tasks)} tasks to compute across {max(1, n_gpus)} GPUs.")

    if n_gpus > 1:
        mp_ctx = mp.get_context("spawn")
        task_queue = mp_ctx.Queue()
        for t in tasks:
            task_queue.put(t)

        workers = []
        num_workers = min(n_gpus, len(tasks))
        for g_id in range(num_workers):
            p = mp_ctx.Process(target=_mp_worker, args=(task_queue, g_id))
            p.start()
            workers.append(p)

        for p in workers:
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f"GPU Worker failed with exitcode {p.exitcode}")
    else:
        if n_gpus == 0 and any(t["method"] == "md" for t in tasks):
            raise RuntimeError("GPU required for interface MD")
        dev = "cuda:0" if n_gpus == 1 else "cpu"
        for t in tasks:
            process_single_task(t, dev)

    finalize_bootstrap(tasks)

if __name__ == "__main__":
    main()
