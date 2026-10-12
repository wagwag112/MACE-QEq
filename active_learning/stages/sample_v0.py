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

    dyn = Langevin(work, timestep=timestep_fs * units.fs, temperature_K=temp_k, friction=0.01 / units.fs)

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
    for sys_name, spec in cfg.V0_SPECS["test"].items():
        total_ps = spec["t_equil_ps"] + spec["n_frames"] * spec["dt_ps"]
        tasks.append({
            "task_id": len(tasks),
            "group": "test",
            "system": sys_name,
            "struct": spec["struct"],
            "n_frames": spec["n_frames"],
            "dt_ps": spec["dt_ps"],
            "t_equil_ps": spec["t_equil_ps"],
            "total_ps": total_ps,
            "traj_file": cfg.V0_TRAJ_DIR / f"{sys_name}_test.traj",
        })
    for sys_name, spec in cfg.V0_SPECS["tv"].items():
        total_target = spec["n_train"] + spec["n_valid"]
        total_ps = spec["t_equil_ps"] + total_target * spec["dt_ps"]
        tasks.append({
            "task_id": len(tasks),
            "group": "tv",
            "system": sys_name,
            "struct": spec["struct"],
            "n_train": spec["n_train"],
            "n_valid": spec["n_valid"],
            "dt_ps": spec["dt_ps"],
            "t_equil_ps": spec["t_equil_ps"],
            "total_ps": total_ps,
            "traj_file": cfg.V0_TRAJ_DIR / f"{sys_name}_tv.traj",
        })
    return tasks

def process_single_task(task, device):
    struct_file = task["struct"]
    if not struct_file.is_file():
        raise FileNotFoundError(f"Structure missing: {struct_file}")
    atoms = read(struct_file)
    traj_file = task["traj_file"]
    
    print(f"[{device}] START Task {task['task_id']}: {task['group']} {task['system']} ({task['total_ps']:.1f} ps)")
    run_md_simulation(
        atoms=atoms,
        model_path=cfg.PRETRAINED_MACE_MODEL,
        output_traj=traj_file,
        total_ps=task["total_ps"],
        timestep_fs=1.0,
        temp_k=getattr(cfg, "TEMPERATURE_K", 300.0),
        save_every=cfg.SAVE_EVERY,
        device=device,
    )
    
    cal_root = cfg.ROOT / "DFT" / "v0" / "cal"
    cal_root.mkdir(parents=True, exist_ok=True)
    sys_name = task["system"]
    manifest_rows = []
    
    if task["group"] == "test":
        n = task["n_frames"]
        selected = extract_sampled_frames(traj_file, n, task["dt_ps"], task["t_equil_ps"], cfg.SAVE_EVERY, 1.0)
        test_frames = []
        for idx, frame in enumerate(selected):
            source = f"test_{sys_name}_{idx:03d}"
            frame.info["source"] = source
            frame.info["system"] = sys_name
            frame.info["al_role"] = "test"
            frame.info["al_round"] = 0
            test_frames.append(frame)
            job_dir = cal_root / sys_name / source
            job_dir.mkdir(parents=True, exist_ok=True)
            write(job_dir / "POSCAR", frame, format="vasp", direct=True, vasp5=True, sort=False)
            manifest_rows.append({
                "round": 0, "parent_round": -1, "system": sys_name,
                "role": "test", "source": source, "filename": f"{source}.vasp",
                "time_ps": frame.info.get("time_ps", 0.0),
            })
        write(cfg.V0_SAMPLED_DIR / f"test_sampled_{sys_name}.extxyz", test_frames, format="extxyz")
        
    elif task["group"] == "tv":
        n_train = task["n_train"]
        n_valid = task["n_valid"]
        total_target = n_train + n_valid
        selected = extract_sampled_frames(traj_file, total_target, task["dt_ps"], task["t_equil_ps"], cfg.SAVE_EVERY, 1.0)
        train_part = selected[:n_train]
        valid_part = selected[n_train:]
        train_frames, valid_frames = [], []
        
        for idx, frame in enumerate(train_part):
            source = f"train_{sys_name}_{idx:03d}"
            frame.info["source"] = source
            frame.info["system"] = sys_name
            frame.info["al_role"] = "train"
            frame.info["al_round"] = 0
            train_frames.append(frame)
            job_dir = cal_root / sys_name / source
            job_dir.mkdir(parents=True, exist_ok=True)
            write(job_dir / "POSCAR", frame, format="vasp", direct=True, vasp5=True, sort=False)
            manifest_rows.append({
                "round": 0, "parent_round": -1, "system": sys_name,
                "role": "train", "source": source, "filename": f"{source}.vasp",
                "time_ps": frame.info.get("time_ps", 0.0),
            })
            
        for idx, frame in enumerate(valid_part):
            source = f"valid_{sys_name}_{idx:03d}"
            frame.info["source"] = source
            frame.info["system"] = sys_name
            frame.info["al_role"] = "valid"
            frame.info["al_round"] = 0
            valid_frames.append(frame)
            job_dir = cal_root / sys_name / source
            job_dir.mkdir(parents=True, exist_ok=True)
            write(job_dir / "POSCAR", frame, format="vasp", direct=True, vasp5=True, sort=False)
            manifest_rows.append({
                "round": 0, "parent_round": -1, "system": sys_name,
                "role": "valid", "source": source, "filename": f"{source}.vasp",
                "time_ps": frame.info.get("time_ps", 0.0),
            })
        write(cfg.V0_SAMPLED_DIR / f"train_sampled_{sys_name}.extxyz", train_frames, format="extxyz")
        write(cfg.V0_SAMPLED_DIR / f"valid_sampled_{sys_name}.extxyz", valid_frames, format="extxyz")

    slice_manifest = cfg.V0_BOOTSTRAP_DIR / f"v0_manifest_task_{task['task_id']}.csv"
    with open(slice_manifest, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["round", "parent_round", "system", "role", "source", "filename", "time_ps"])
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"[{device}] TASK {task['task_id']} ({task['group']} {task['system']}) COMPLETE -> POSCARs written directly to ROM.")

def finalize_bootstrap(tasks):
    cal_root = cfg.ROOT / "DFT" / "v0" / "cal"
    manifest_file = cfg.V0_BOOTSTRAP_DIR / "v0_selection_manifest.csv"
    
    all_manifest_rows = []
    for t in tasks:
        slice_file = cfg.V0_BOOTSTRAP_DIR / f"v0_manifest_task_{t['task_id']}.csv"
        if not slice_file.is_file():
            return False
        with open(slice_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            all_manifest_rows.extend(list(reader))
            
    if len(all_manifest_rows) != cfg.V0_TARGET_COUNT:
        raise RuntimeError(f"Manifest target mismatch: {len(all_manifest_rows)} != {cfg.V0_TARGET_COUNT}")
        
    with open(manifest_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["round", "parent_round", "system", "role", "source", "filename", "time_ps"])
        writer.writeheader()
        writer.writerows(all_manifest_rows)
        
    for role in ["test", "train", "valid"]:
        role_frames = []
        for p in sorted(cfg.V0_SAMPLED_DIR.glob(f"{role}_sampled_*.extxyz")):
            role_frames.extend(read(p, index=":"))
        write(cfg.V0_SAMPLED_DIR / f"{role}_sampled.extxyz", role_frames, format="extxyz")
        for p in cfg.V0_SAMPLED_DIR.glob(f"{role}_sampled_*.extxyz"):
            p.unlink()

    for t in tasks:
        slice_file = cfg.V0_BOOTSTRAP_DIR / f"v0_manifest_task_{t['task_id']}.csv"
        if slice_file.is_file():
            slice_file.unlink()

    poscars = list(cal_root.rglob("POSCAR"))
    print(f"SAMPLE V0 COMPLETE: POSCAR_jobs={len(poscars)} manifest={manifest_file}")
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

    # Array task support (neu chay Slurm Array)
    if "SLURM_ARRAY_TASK_ID" in os.environ:
        task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
        process_single_task(tasks[task_id], "cuda:0")
        finalize_bootstrap(tasks)
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
        dev = "cuda:0" if n_gpus == 1 else "cpu"
        for t in tasks:
            process_single_task(t, dev)

    finalize_bootstrap(tasks)

if __name__ == "__main__":
    main()
