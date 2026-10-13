from __future__ import annotations
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
AL_DIR = SCRIPT_DIR.parent
if str(AL_DIR) not in sys.path:
    sys.path.insert(0, str(AL_DIR))

import al_config as cfg

CHECK_INTERVAL_SEC = 300
MAX_RETRIES = 5
BLOCKING_REASONS = {"Priority", "Resources", "ReqNodeNotAvail", "Nodes_required_for_job_are_DOWN_or_DRAINED"}
CORRUPT_FILES = [
    "WAVECAR", "CHGCAR", "CHG", "vasprun.xml", "vasp.out", "vasp.err",
    "OSZICAR", "OUTCAR", "REPORT", "XDATCAR", "DOSCAR", "EIGENVAL",
    "PCDAT", "IBZKPT", ".DONE"
]

def clean_calc_dir(calc_dir: Path) -> None:
    for fname in CORRUPT_FILES:
        fpath = calc_dir / fname
        if fpath.exists():
            try:
                fpath.unlink()
            except Exception:
                pass

def get_cal_dirs() -> list[Path]:
    cal_root = cfg.ROOT / "DFT" / f"v{cfg.ROUND}" / "cal"
    if not cal_root.is_dir():
        raise FileNotFoundError(f"Missing DFT calculation root: {cal_root}")
    dirs = [p.parent for p in sorted(cal_root.rglob("POSCAR"))]
    target_count = getattr(cfg, "V0_TARGET_COUNT", 400) if cfg.ROUND == 0 else cfg.N_SELECTED
    if len(dirs) != target_count:
        raise RuntimeError(f"Expected {target_count} calculation directories, found {len(dirs)}")
    return dirs

def is_vasprun_valid(calc_dir: Path) -> bool:
    xml = calc_dir / "vasprun.xml"
    if not xml.is_file() or xml.stat().st_size == 0:
        return False
    try:
        with open(xml, "rb") as f:
            f.seek(-500, os.SEEK_END)
            return b"</modeling>" in f.read()
    except Exception:
        return False

def submit_vasp_job(calc_dir: Path, partition: str) -> str:
    clean_calc_dir(calc_dir)
    system = calc_dir.parent.name
    tmpl_dir = cfg.DFT_TEMPLATE_DIR / system
    if not tmpl_dir.is_dir():
        raise FileNotFoundError(f"Missing DFT template directory for system '{system}': {tmpl_dir}")

    for f in ["INCAR", "KPOINTS", "POTCAR"]:
        src, dst = tmpl_dir / f, calc_dir / f
        if not dst.exists() and src.exists():
            subprocess.run(["cp", str(src), str(dst)], check=True)

    run_sh = cfg.DFT_RUN_SCRIPT
    if not run_sh.is_file():
        raise FileNotFoundError(f"Missing DFT run script: {run_sh}")
    dst_run = calc_dir / "run.sh"
    if not dst_run.exists():
        subprocess.run(["cp", str(run_sh), str(dst_run)], check=True)
        dst_run.chmod(0o755)

    slurm_script = f"""#!/bin/bash
#SBATCH --job-name=v{cfg.ROUND}_{calc_dir.name[:10]}
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --ntasks={cfg.DFT_CPUS_PER_JOB}
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G
#SBATCH --time={cfg.DFT_TIME_LIMIT}
#SBATCH --output={calc_dir}/vasp.out
#SBATCH --error={calc_dir}/vasp.err

cd {calc_dir}
bash run.sh
"""
    res = subprocess.run(["sbatch", "--parsable"], input=slurm_script, text=True, capture_output=True, check=True)
    return res.stdout.strip().split(";")[0]

def query_slurm(job_ids: list[str]):
    if not job_ids:
        return {}
    res = subprocess.run(["squeue", "--me", "-h", "-o", "%i|%T|%r"], text=True, capture_output=True)
    if res.returncode != 0:
        return None
    status_map = {}
    if res.stdout.strip():
        for line in res.stdout.strip().splitlines():
            parts = line.split("|")
            if len(parts) >= 2:
                status_map[parts[0].strip()] = (parts[1].strip(), parts[2].strip() if len(parts) > 2 else "")
    return status_map

def main():
    if cfg.ROUND != 0:
        import stages.prepare_dft as prep
        rows = prep.load_manifest()
        prep.validate_manifest(rows)
        prep.validate_sources(rows)
        prep.prepare(rows)
    partitions = cfg.DFT_PARTITIONS
    n_part = len(partitions)
    calc_dirs = get_cal_dirs()
    tracking_file = cfg.ACTIVE_LEARNING_DIR / "state" / f"dft_v{cfg.ROUND}_jobs.json"
    tracking = {}
    if tracking_file.is_file():
        with open(tracking_file, "r") as f:
            tracking = json.load(f)

    unsubmitted = [d for d in calc_dirs if str(d) not in tracking and not is_vasprun_valid(d)]
    if unsubmitted:
        print(f"=== Initial Dispatch: Submitting {len(unsubmitted)} jobs across {n_part} partitions ===")
        for idx, cdir in enumerate(unsubmitted):
            assigned_part = partitions[idx % n_part]
            jid = submit_vasp_job(cdir, assigned_part)
            tracking[str(cdir)] = {
                "job_id": jid,
                "partition": assigned_part,
                "submit_time": time.time(),
                "retries": 0,
            }
        tracking_file.parent.mkdir(parents=True, exist_ok=True)
        with open(tracking_file, "w") as f:
            json.dump(tracking, f, indent=2)

    while True:
        completed = sum(1 for d in calc_dirs if is_vasprun_valid(d))
        active_dirs = {info["job_id"]: p for p, info in tracking.items() if not is_vasprun_valid(Path(p))}
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Progress: {completed}/{len(calc_dirs)} completed. Active jobs: {len(active_dirs)}")

        if completed == len(calc_dirs):
            print("ALL DFT JOBS COMPLETED SUCCESSFULLY.")
            break

        slurm_statuses = query_slurm(list(active_dirs.keys()))
        if slurm_statuses is None:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [WARN] squeue query error; skipping cycle.")
            time.sleep(30)
            continue
        now = time.time()
        updated = False

        for jid, sdir in active_dirs.items():
            info = tracking[sdir]
            st, reason = slurm_statuses.get(jid, ("UNKNOWN", ""))
            calc_path = Path(sdir)

            if st == "PENDING":
                elapsed = now - info["submit_time"]
                is_blocked = any(br in reason for br in BLOCKING_REASONS)
                if is_blocked and elapsed >= 1800:
                    curr_part = info["partition"]
                    next_part = partitions[(partitions.index(curr_part) + 1) % n_part]
                    up = subprocess.run(["scontrol", "update", f"JobId={jid}", f"Partition={next_part}"], capture_output=True, text=True)
                    if up.returncode == 0:
                        print(f"[RE-ROUTE] Job {jid} ({calc_path.name}): {curr_part} -> {next_part} (Reason: {reason})")
                        info["partition"] = next_part
                        info["submit_time"] = now
                        info["reroutes"] = info.get("reroutes", 0) + 1
                        updated = True

            elif st in {"UNKNOWN", "FAILED", "TIMEOUT", "CANCELLED", "NODE_FAIL"}:
                elapsed = now - info["submit_time"]
                if st == "UNKNOWN" and elapsed < 180:
                    continue
                if not is_vasprun_valid(calc_path) and info["retries"] < MAX_RETRIES:
                    curr_part = info["partition"]
                    next_part = partitions[(partitions.index(curr_part) + 1) % n_part]
                    new_jid = submit_vasp_job(calc_path, next_part)
                    print(f"[RE-SUBMIT] Job {jid} ({calc_path.name}) failed or timed out. Resubmitted as {new_jid} on {next_part} (12h walltime)")
                    info["job_id"] = new_jid
                    info["partition"] = next_part
                    info["submit_time"] = now
                    info["retries"] += 1
                    updated = True

        if updated:
            with open(tracking_file, "w") as f:
                json.dump(tracking, f, indent=2)

        time.sleep(CHECK_INTERVAL_SEC)

if __name__ == "__main__":
    main()
