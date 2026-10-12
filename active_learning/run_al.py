"""
Canonical Slurm orchestrator for the MACE-QEq active-learning workflow.

Public commands
---------------
python run_al.py --status
python run_al.py --run-round

Workflow
--------
exploration
    -> committee_scoring
    -> candidate_selection
    -> prepare_dft
    -> dft (automated dispatcher)
    -> collect_dft
    -> qeq_residual
    -> merge_dataset
    -> train_models
    -> train_committee
    -> evaluation
    -> fresh_probe
    -> probe_scoring
    -> convergence_report
    -> STOP FOR SCIENTIFIC REVIEW

Policy
------
- DFT execution is fully automated via dft_dispatch.slurm.
- Completed scientific artifacts are not regenerated.
- Active Slurm jobs may be reused.
- Historical completed/failed job IDs do not replace artifact validation.
- The next active-learning generation is never started automatically.
- The convergence report is a decision aid requiring scientific review.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys

from datetime import datetime
from pathlib import Path

from ase.io import read

import al_config as cfg
import importlib

def sync_config_round(target_round: int) -> None:
    parent_round = -1 if target_round == 0 else target_round - 1
    cfg_path = AL_DIR / "al_config.py"
    content = cfg_path.read_text(encoding="utf-8")

    new_content = re.sub(r"^ROUND\s*=.*", f"ROUND = {target_round}", content, flags=re.MULTILINE)
    new_content = re.sub(r"^PARENT_ROUND\s*=.*", f"PARENT_ROUND = {parent_round}", new_content, flags=re.MULTILINE)

    if new_content != content:
        cfg_path.write_text(new_content, encoding="utf-8")

    importlib.reload(cfg)
    global STATE_FILE, JOB_FILE
    STATE_FILE = STATE_DIR / f"round_v{cfg.ROUND}.json"
    JOB_FILE = STATE_DIR / f"jobs_v{cfg.ROUND}.json"

def detect_current_round() -> int:
    r = 0
    while True:
        conv_json = (
            cfg.MD_ROOT
            / f"active_learning_v{r}"
            / "validation"
            / "convergence"
            / "convergence_report.json"
        )
        if not conv_json.is_file():
            return r
        try:
            with conv_json.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            rec = data.get("recommendation", "")
            if rec == "CONTINUE_AL":
                r += 1
            else:
                return r
        except Exception:
            return r



# ============================================================================
# DIRECTORIES
# ============================================================================

AL_DIR = Path(__file__).resolve().parent

STATE_DIR = (
    AL_DIR
    / "state"
)

LOG_ROOT = (
    AL_DIR
    / "logs"
)

STATE_FILE = (
    STATE_DIR
    / f"round_v{cfg.ROUND}.json"
)

JOB_FILE = (
    STATE_DIR
    / f"jobs_v{cfg.ROUND}.json"
)
# ============================================================================
# WORKFLOW
# ============================================================================

STAGES = [
    "sample_v0",
    "committee_scoring",
    "candidate_selection",
    "dft",
    "collect_dft",
    "qeq_residual",
    "merge_dataset",
    "train_models",
    "train_committee",
    "evaluation",
    "fresh_probe",
    "probe_scoring",
    "convergence_report",
]


SLURM_SCRIPTS = {
    "sample_v0":
        "slurm/sample_v0.slurm",

    "committee_scoring":
        "slurm/committee_scoring.slurm",

    "dft":
        "slurm/dft_dispatch.slurm",

    "candidate_selection":
        "slurm/cpu_stage.slurm",

    "collect_dft":
        "slurm/cpu_stage.slurm",

    "qeq_residual":
        "slurm/qeq_residual.slurm",


    "merge_dataset":
        "slurm/cpu_stage.slurm",

    "train_models":
        "slurm/train_models.slurm",

    "train_committee":
        "slurm/train_committee.slurm",

    "evaluation":
        "slurm/evaluation.slurm",

    "fresh_probe":
        "slurm/fresh_probe.slurm",

    "probe_scoring":
        "slurm/probe_scoring.slurm",

    "convergence_report":
        "slurm/convergence_report.slurm",
}


CPU_STAGES = {
    "candidate_selection",
    "collect_dft",
    "merge_dataset",
}


ACTIVE_SLURM_STATES = {
    "RUNNING",
    "PENDING",
    "CONFIGURING",
    "COMPLETING",
}


FAILED_SLURM_STATES = {
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "NODE_FAIL",
    "PREEMPTED",
    "BOOT_FAIL",
    "DEADLINE",
}


# ============================================================================
# GENERAL UTILITIES
# ============================================================================

def now_string():

    return datetime.now().isoformat(
        timespec="seconds"
    )


def ensure_directories():

    STATE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for stage in STAGES:

        (
            LOG_ROOT
            / f"v{cfg.ROUND}"
            / stage
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

def load_json(
    path,
    default,
):

    path = Path(
        path
    )

    if not path.exists():

        return default.copy()

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:

        return json.load(
            handle
        )


def save_json(
    path,
    data,
):

    path = Path(
        path
    )

    tmp = path.with_suffix(
        path.suffix
        + ".tmp"
    )

    with tmp.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            data,
            handle,
            indent=2,
            sort_keys=True,
        )

    tmp.replace(
        path
    )


def initialize_state():

    if not STATE_FILE.exists():

        save_json(
            STATE_FILE,
            {
                "round":
                    cfg.ROUND,

                "parent_round":
                    cfg.PARENT_ROUND,

                "created":
                    now_string(),

                "updated":
                    now_string(),

                "stages":
                    {},
            },
        )

    if not JOB_FILE.exists():

        save_json(
            JOB_FILE,
            {
                "round":
                    cfg.ROUND,

                "jobs":
                    {},
            },
        )


def file_ok(
    path,
):

    path = Path(
        path
    )

    return (
        path.is_file()
        and path.stat().st_size > 0
    )


# ============================================================================
# MODEL VALIDATION
# ============================================================================

def exact_model_exists(
    directory,
    model_name,
    seed,
):

    model_file = (
        Path(
            directory
        )
        / "checkpoints"
        / f"{model_name}_run-{seed}.model"
    )

    return file_ok(
        model_file
    )


def sample_v0_complete():
    if main_models_complete():
        return True
    cal_root = cfg.ROOT / "DFT" / "v0" / "cal"
    if not cal_root.is_dir():
        return False
    poscars = list(cal_root.rglob("POSCAR"))
    target = getattr(cfg, "V0_TARGET_COUNT", 400)
    manifest = getattr(cfg, "V0_BOOTSTRAP_DIR", cfg.MD_ROOT / "v0_bootstrap") / "v0_selection_manifest.csv"
    return len(poscars) == target and file_ok(manifest)


# ============================================================================
# EXPLORATION VALIDATION
# ============================================================================


# ============================================================================
# DFT VALIDATION
# ============================================================================

def dft_complete():
    """
    Return True only when every structure selected for the current
    round has exactly one valid, parseable vasprun.xml.
    """
    if cfg.ROUND == 0 and main_models_complete():
        return True

    try:

        from stages.collect_dft import (
            load_expected_sources,
            locate_selected_vaspruns,
            read_vasprun,
        )

        expected_sources = (
            load_expected_sources()
        )

        xml_files = (
            locate_selected_vaspruns(
                expected_sources
            )
        )

    except Exception:

        return False

    expected_count = getattr(cfg, "V0_TARGET_COUNT", 400) if cfg.ROUND == 0 else cfg.N_SELECTED
    if len(xml_files) != expected_count:
        return False

    for xml_file in xml_files:
        if not file_ok(xml_file):
            return False
        try:
            with open(xml_file, "rb") as f:
                f.seek(-500, 2)
                if b"</modeling>" not in f.read():
                    return False
        except Exception:
            return False
    return True


# ============================================================================
# MODEL VALIDATION
# ============================================================================

def main_models_complete():

    full_ok = exact_model_exists(
        cfg.TRAIN_FULL_DIR,
        cfg.MODEL_FULL_NAME,
        cfg.BASE_SEED,
    )

    short_ok = exact_model_exists(
        cfg.TRAIN_SHORT_DIR,
        cfg.MODEL_SHORT_NAME,
        cfg.BASE_SEED,
    )

    return (
        full_ok
        and short_ok
    )


def committee_complete():

    seeds = [
        seed
        for seed in cfg.COMMITTEE_SEEDS
        if seed != cfg.BASE_SEED
    ]

    found = 0

    for seed in seeds:

        model_name = (
            f"MACE_short_v{cfg.ROUND}_"
            f"seed{seed}"
        )

        model_dir = (
            cfg.COMMITTEE_NEXT_DIR
            / f"seed_{seed}"
        )

        if exact_model_exists(
            model_dir,
            model_name,
            seed,
        ):

            found += 1

    return (
        found == len(
            seeds
        ),
        found,
        len(
            seeds
        ),
    )


# ============================================================================
# POST-TRAINING VALIDATION
# ============================================================================

def evaluation_complete():

    required = [
        cfg.EVALUATION_SUMMARY,
        cfg.EVALUATION_FRAMES,
    ]

    return all(
        file_ok(
            path
        )
        for path in required
    )


def fresh_probe_complete():
    if not file_ok(cfg.PROBE_TRAJECTORY_FILE):
        return False
    try:
        with open(cfg.PROBE_TRAJECTORY_FILE, "rb") as f:
            count = sum(1 for line in f if b"Lattice=" in line)
        return abs(count - cfg.PROBE_EXPECTED_MD_FRAMES) <= 2 and count > 0
    except Exception:
        return False

def probe_scoring_complete():

    return (
        file_ok(
            cfg.PROBE_SCORE_FILE
        )
        and
        file_ok(
            cfg.PROBE_TOP_UNCERTAINTY_FILE
        )
    )


def convergence_report_complete():

    return (
        file_ok(
            cfg.CONVERGENCE_REPORT
        )
        and
        file_ok(
            cfg.CONVERGENCE_JSON
        )
    )


def convergence_recommendation():

    if not file_ok(
        cfg.CONVERGENCE_JSON
    ):

        return None

    try:

        with Path(
            cfg.CONVERGENCE_JSON
        ).open(
            "r",
            encoding="utf-8",
        ) as handle:

            data = json.load(
                handle
            )

    except Exception:

        return None

    value = data.get(
        "recommendation"
    )

    if value is None:

        return None

    return str(
        value
    )


# ============================================================================
# OUTPUT VALIDATION
# ============================================================================

def stage_output_status(
    stage,
):
    if stage == "sample_v0":
        if cfg.ROUND != 0 or main_models_complete():
            return ("DONE", "Bootstrap baseline generation complete.")
        if sample_v0_complete():
            return ("DONE", f"Bootstrap 400 structures generated.")
        return ("PENDING", "Bootstrap sampling incomplete.")

    if cfg.ROUND == 0 and stage in {"committee_scoring", "candidate_selection"}:
        return ("DONE", "Bypassed in v0 bootstrap.")

    if cfg.ROUND == 0 and main_models_complete() and stage in {"dft", "collect_dft", "qeq_residual"}:
        return ("DONE", "Bootstrap baseline datasets integrated into trained models.")



    if stage == "committee_scoring":
        if file_ok(cfg.SCORE_FILE):
            return (
                "DONE",
                "Committee score file detected.",
            )
        if cfg.PARENT_ROUND >= 0 and cfg.SELECTION_TRAJECTORY_FILE == cfg.PREV_PROBE_TRAJ:
            prev_scoring_dir = (
                cfg.MD_ROOT
                / f"active_learning_v{cfg.PARENT_ROUND}"
                / "validation"
                / "fresh_probe_scoring"
            )
            prev_score = prev_scoring_dir / "committee_scores.csv"
            if file_ok(prev_score):
                cfg.SCORING_DIR.mkdir(parents=True, exist_ok=True)
                cfg.SCORING_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
                shutil.copy2(prev_score, cfg.SCORE_FILE)
                prev_top50 = prev_scoring_dir / "top50_uncertainty.csv"
                if file_ok(prev_top50):
                    shutil.copy2(prev_top50, cfg.TOP_UNCERTAINTY_FILE)
                prev_fig = prev_scoring_dir / "figures" / "sigma_p95_vs_time.png"
                if file_ok(prev_fig):
                    shutil.copy2(prev_fig, cfg.SCORING_FIGURE_DIR / "sigma_p95_vs_time.png")
                return (
                    "DONE",
                    f"Inherited committee scores from v{cfg.PARENT_ROUND} fresh probe.",
                )
        return (
            "PENDING",
            "Committee score file missing.",
        )

    if stage == "candidate_selection":

        if file_ok(
            cfg.SELECTION_MANIFEST
        ):

            return (
                "DONE",
                "Selection manifest detected.",
            )

        return (
            "PENDING",
            "Selection manifest missing.",
        )

    if stage == "dft":

        if dft_complete():

            return (
                "DONE",
                (
                    f"All {cfg.N_SELECTED} "
                    "DFT calculations completed."
                ),
            )

        return (
            "PENDING",
            "DFT calculations incomplete.",
        )

    if stage == "collect_dft":

        if (
            file_ok(
                cfg.DFT_FILE
            )
            and
            file_ok(
                cfg.DFT_REPORT
            )
        ):

            return (
                "DONE",
                "Collected DFT dataset detected.",
            )

        return (
            "PENDING",
            "Collected DFT output missing.",
        )

    if stage == "qeq_residual":

        if (
            file_ok(
                cfg.SHORT_FILE
            )
            and
            file_ok(
                cfg.QEQ_REPORT
            )
        ):

            return (
                "DONE",
                "QEq residual dataset detected.",
            )

        return (
            "PENDING",
            "QEq residual output missing.",
        )


    if stage == "merge_dataset":
        outputs = [
            cfg.OUT_TRAIN_DFT,
            cfg.OUT_TRAIN_SHORT,
            cfg.OUT_VALID_DFT,
            cfg.OUT_VALID_SHORT,
        ]
        if cfg.ROUND > 0:
            outputs.extend([
                cfg.ROUND_TRAIN_DFT,
                cfg.ROUND_TRAIN_SHORT,
                cfg.ROUND_HOLDOUT_DFT,
                cfg.ROUND_HOLDOUT_SHORT,
            ])
        if all(file_ok(path) for path in outputs):
            return ("DONE", "Merged generation dataset detected.")
        return ("PENDING", "Merged generation dataset incomplete.")

    if stage == "train_models":

        if main_models_complete():

            return (
                "DONE",
                "Final full and short models detected.",
            )

        return (
            "PENDING",
            "Principal model training incomplete.",
        )

    if stage == "train_committee":

        (
            ok,
            found,
            expected,
        ) = committee_complete()

        if ok:

            return (
                "DONE",
                (
                    "Committee training complete "
                    f"({found}/{expected})."
                ),
            )

        return (
            "PENDING",
            (
                "Committee training incomplete "
                f"({found}/{expected})."
            ),
        )

    # ------------------------------------------------------------------
    # Post-training evaluation
    # ------------------------------------------------------------------

    if stage == "evaluation":

        if evaluation_complete():

            return (
                "DONE",
                "Model evaluation outputs detected.",
            )

        return (
            "PENDING",
            "Model evaluation incomplete.",
        )

    # ------------------------------------------------------------------
    # Fresh current-generation MD probe
    # ------------------------------------------------------------------

    if stage == "fresh_probe":

        if fresh_probe_complete():

            return (
                "DONE",
                (
                    "Fresh convergence probe complete "
                    f"({cfg.PROBE_EXPECTED_MD_FRAMES} frames)."
                ),
            )

        return (
            "PENDING",
            "Fresh convergence probe incomplete.",
        )

    # ------------------------------------------------------------------
    # Fresh-probe committee scoring
    # ------------------------------------------------------------------

    if stage == "probe_scoring":

        if probe_scoring_complete():

            return (
                "DONE",
                "Fresh-probe committee scores detected.",
            )

        return (
            "PENDING",
            "Fresh-probe committee scoring incomplete.",
        )

    # ------------------------------------------------------------------
    # Convergence report
    # ------------------------------------------------------------------

    if stage == "convergence_report":

        if convergence_report_complete():

            recommendation = (
                convergence_recommendation()
            )

            if recommendation is None:

                text = (
                    "Convergence report detected."
                )

            else:

                text = (
                    "Convergence report complete "
                    f"({recommendation})."
                )

            return (
                "DONE",
                text,
            )

        return (
            "PENDING",
            "Convergence report missing.",
        )

    raise RuntimeError(
        f"Unknown stage: {stage}"
    )


# ============================================================================
# SLURM
# ============================================================================

def command_output(
    command,
):

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )

    return (
        result.returncode,
        result.stdout.strip(),
        result.stderr.strip(),
    )


def slurm_state(
    job_id,
):

    job_id = str(
        job_id
    )

    # ------------------------------------------------------------------
    # Active jobs from squeue
    # ------------------------------------------------------------------

    code, stdout, _ = (
        command_output(
            [
                "squeue",
                "-h",
                "-j",
                job_id,
                "-o",
                "%T",
            ]
        )
    )

    if (
        code == 0
        and stdout
    ):

        states = {
            line.strip().upper()
            for line in stdout.splitlines()
            if line.strip()
        }

        if "RUNNING" in states:

            return "RUNNING"

        if states & ACTIVE_SLURM_STATES:

            return "PENDING"

    # ------------------------------------------------------------------
    # Historical jobs from sacct
    # ------------------------------------------------------------------

    code, stdout, _ = (
        command_output(
            [
                "sacct",
                "-n",
                "-X",
                "-j",
                job_id,
                "--format=State",
            ]
        )
    )

    if (
        code != 0
        or not stdout
    ):

        return "UNKNOWN"

    states = []

    for line in stdout.splitlines():

        text = line.strip()

        if not text:

            continue

        state = (
            text
            .split()[0]
            .split("+")[0]
            .upper()
        )

        states.append(
            state
        )

    if any(
        state in FAILED_SLURM_STATES
        for state in states
    ):

        return "FAILED"

    if any(
        state == "RUNNING"
        for state in states
    ):

        return "RUNNING"

    if any(
        state in ACTIVE_SLURM_STATES
        for state in states
    ):

        return "PENDING"

    if (
        states
        and all(
            state == "COMPLETED"
            for state in states
        )
    ):

        return "COMPLETED"

    if any(
        state == "COMPLETED"
        for state in states
    ):

        return "COMPLETED"

    return "UNKNOWN"


# ============================================================================
# JOB DATABASE
# ============================================================================

def load_jobs():

    return load_json(
        JOB_FILE,
        {
            "round":
                cfg.ROUND,

            "jobs":
                {},
        },
    )


def get_job(
    stage,
):

    jobs = load_jobs()

    return (
        jobs
        .get(
            "jobs",
            {}
        )
        .get(
            stage
        )
    )


def record_job(
    stage,
    job_id,
    script,
    dependency=None,
):

    jobs = load_jobs()

    jobs.setdefault(
        "jobs",
        {},
    )

    jobs[
        "jobs"
    ][stage] = {
        "job_id":
            str(
                job_id
            ),

        "script":
            str(
                script
            ),

        "dependency":
            (
                str(
                    dependency
                )
                if dependency is not None
                else None
            ),

        "submitted":
            now_string(),
    }

    save_json(
        JOB_FILE,
        jobs,
    )


def existing_job_id(
    stage,
):

    job = get_job(
        stage
    )

    if job is None:

        return None

    job_id = job.get(
        "job_id"
    )

    if not job_id:

        return None

    return str(
        job_id
    )


def reusable_job(
    stage,
):

    job_id = existing_job_id(
        stage
    )

    if job_id is None:

        return None

    state = slurm_state(
        job_id
    )

    if state == "RUNNING":

        return job_id

    if state == "PENDING":

        result = subprocess.run(
            [
                "squeue",
                "-h",
                "-j",
                str(job_id),
                "-o",
                "%r",
            ],
            capture_output=True,
            text=True,
        )

        reasons = result.stdout.strip()

        if "DependencyNeverSatisfied" in reasons:

            return None

        return job_id

    return None


# ============================================================================
# SLURM SUBMISSION
# ============================================================================

def submit_slurm(
    stage,
    dependency=None,
):

    script_name = (
        SLURM_SCRIPTS[
            stage
        ]
    )

    script = (
        AL_DIR
        / script_name
    )

    if not script.is_file():

        raise RuntimeError(
            f"Slurm script not found: {script}"
        )

    existing = reusable_job(stage)
    recorded_dep = get_job(stage).get("dependency") if get_job(stage) else None
    expected_dep = str(dependency) if dependency is not None else None

    if existing is not None and recorded_dep == expected_dep:
        print(f"{stage}: reusing job {existing}")
        return existing
    elif existing is not None:
        subprocess.run(["scancel", str(existing)], capture_output=True)

    command = [
        "sbatch",
        "--parsable",
    ]

    if dependency is not None:
        deps = [str(d).strip() for d in (dependency if isinstance(dependency, (list, tuple)) else re.split(r"[:,]", str(dependency))) if str(d).strip()]
        dep_clause = ",".join(f"afterok:{d}" for d in deps)
        command.append(f"--dependency={dep_clause}")

    if stage in CPU_STAGES:

        command.append(
            f"--export=ALL,AL_STAGE={stage}"
        )

    if stage == "train_committee":

        seeds = [
            seed
            for seed in cfg.COMMITTEE_SEEDS
            if seed != cfg.BASE_SEED
        ]

        if not seeds:

            raise RuntimeError(
                "No non-base committee seeds configured."
            )

        command.append(
            f"--array=0-{len(seeds)-1}"
        )

    command.append(
        str(
            script
        )
    )

    result = subprocess.run(
        command,
        cwd=AL_DIR,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:

        raise RuntimeError(
            result.stderr.strip()
        )

    job_id = (
        result.stdout
        .strip()
        .split(";")[0]
    )

    if not job_id:

        raise RuntimeError(
            f"No job ID returned for stage {stage}."
        )

    record_job(
        stage=stage,
        job_id=job_id,
        script=script,
        dependency=dependency,
    )

    print(
        f"Submitted {stage}: "
        f"{job_id}"
    )

    if dependency is not None:

        print(
            f"  afterok: {dependency}"
        )

    return job_id


# ============================================================================
# STATUS
# ============================================================================

def detect_stage(
    stage,
):

    (
        output_status,
        message,
    ) = stage_output_status(
        stage
    )

    if output_status == "DONE":

        return (
            "DONE",
            message,
        )

    job_id = existing_job_id(
        stage
    )

    if job_id is not None:

        state = slurm_state(
            job_id
        )

        if state in {
            "RUNNING",
            "PENDING",
        }:

            return (
                state,
                (
                    f"Slurm job {job_id} "
                    f"is {state.lower()}."
                ),
            )

        if state == "FAILED":

            return (
                "FAILED",
                (
                    f"Slurm job {job_id} failed."
                ),
            )

        if (
            state == "COMPLETED"
            and output_status != "DONE"
        ):

            return (
                "FAILED",
                (
                    f"Slurm job {job_id} completed "
                    "but expected output is missing."
                ),
            )

    return (
        output_status,
        message,
    )


def refresh_state():

    state = load_json(
        STATE_FILE,
        {},
    )

    if (
        state.get(
            "round"
        )
        != cfg.ROUND
    ):

        raise RuntimeError(
            "State-file round differs "
            "from al_config.py."
        )

    result = {}

    for stage in STAGES:

        (
            status,
            message,
        ) = detect_stage(
            stage
        )

        result[
            stage
        ] = {
            "status":
                status,

            "message":
                message,

            "checked":
                now_string(),
        }

    state[
        "stages"
    ] = result

    state[
        "updated"
    ] = now_string()

    save_json(
        STATE_FILE,
        state,
    )

    return state


def print_scientific_summary():

    if not convergence_report_complete():

        return

    try:

        with Path(
            cfg.CONVERGENCE_JSON
        ).open(
            "r",
            encoding="utf-8",
        ) as handle:

            data = json.load(
                handle
            )

    except Exception:

        return

    recommendation = (
        data.get(
            "recommendation",
            "UNKNOWN",
        )
    )

    probe = (
        data.get(
            "fresh_probe",
            {}
        )
    )

    test = (
        data.get(
            "evaluation",
            {}
        ).get(
            "independent_test",
            {}
        )
    )

    print()
    print("=" * 86)
    print(
        "SCIENTIFIC CONVERGENCE SUMMARY"
    )
    print("=" * 86)

    print(
        f"Recommendation     : {recommendation}"
    )

    f_metric = test.get("current_shortQEq_F_RMSE_eV_A", test.get("F_RMSE_shortQEq_eV_A"))
    if f_metric is not None:
        print(
            "Independent test F : "
            f"{float(f_metric):.6f} "
            "eV/A (short+QEq)"
        )

    if (
        "sigma_p95_mean_eV_A"
        in probe
    ):

        print(
            "Fresh sigma mean   : "
            f"{probe['sigma_p95_mean_eV_A']:.6f} "
            "eV/A"
        )

    if (
        "sigma_p95_distribution_p95_eV_A"
        in probe
    ):

        print(
            "Fresh sigma p95    : "
            f"{probe['sigma_p95_distribution_p95_eV_A']:.6f} "
            "eV/A"
        )

    if (
        "fraction_ge_target"
        in probe
    ):

        print(
            "Frames >= target   : "
            f"{100.0 * probe['fraction_ge_target']:.3f}%"
        )

    if (
        "fraction_ge_hard"
        in probe
    ):

        print(
            "Frames >= hard     : "
            f"{100.0 * probe['fraction_ge_hard']:.3f}%"
        )

    print()
    print(
        "The workflow does NOT start another "
        "active-learning round automatically."
    )

    print(
        "Review the convergence report before "
        "changing ROUND/PARENT_ROUND."
    )

    print("=" * 86)


def print_status(
    state,
):

    print()
    print("=" * 86)

    print(
        f"ACTIVE LEARNING ROUND v{cfg.ROUND}"
    )

    print(
        f"Parent round       : v{cfg.PARENT_ROUND}"
    )

    print(
        f"System             : {cfg.SYSTEM}"
    )

    print(
        f"Temperature        : "
        f"{cfg.TEMPERATURE_K} K"
    )

    print(
        f"Exploration MD     : "
        f"{cfg.MD_PS:g} ps"
    )

    print(
        f"Fresh probe MD     : "
        f"{cfg.PROBE_MD_PS:g} ps"
    )

    print("=" * 86)

    for stage in STAGES:

        entry = (
            state[
                "stages"
            ][stage]
        )

        print(
            f"{stage:22s} "
            f"{entry['status']:10s} "
            f"{entry['message']}"
        )

    print("=" * 86)

    print(
        f"State : {STATE_FILE}"
    )

    print(
        f"Jobs  : {JOB_FILE}"
    )

    print_scientific_summary()

    print()


# ============================================================================
# MANUAL DFT GATE
# ============================================================================

def dft_inputs_prepared():
    """
    Return True only when the manual-DFT hand-off directory contains
    exactly the current-round selected structures.

    VASP execution itself is intentionally external to the AL pipeline.
    """

    try:
        from stages.prepare_dft import (
            load_manifest,
            validate_manifest,
            output_complete,
        )

        rows = load_manifest()
        validate_manifest(rows)

    except Exception:
        return False

    return output_complete(rows)


def run_prepare_dft():
    if not file_ok(cfg.SELECTION_MANIFEST):
        return

    if dft_inputs_prepared():

        print(
            "prepare_dft: already complete"
        )

        return

    script = (
        AL_DIR
        / "stages"
        / "prepare_dft.py"
    )

    if not file_ok(script):

        raise RuntimeError(
            f"prepare_dft.py not found or empty: {script}"
        )

    print()
    print("=" * 78)
    print("PREPARE MANUAL DFT INPUTS")
    print("=" * 78)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "stages.prepare_dft",
        ],
        cwd=AL_DIR,
    )

    if result.returncode != 0:

        raise RuntimeError(
            "prepare_dft.py failed."
        )

    if not dft_inputs_prepared():

        raise RuntimeError(
            "prepare_dft.py returned successfully, "
            "but the prepared DFT input set is incomplete."
        )

    print()
    print(
        f"Prepared {cfg.N_SELECTED} DFT structures."
    )


def launch_post_dft_chain(dependency=None):
    prep_stages = ["collect_dft", "qeq_residual", "merge_dataset"]
    last_prep_job = dependency
    for stage in prep_stages:
        output_status, _ = stage_output_status(stage)
        if output_status == "DONE":
            last_prep_job = None
            continue
        existing = reusable_job(stage)
        if existing is not None:
            last_prep_job = existing
            continue
        last_prep_job = submit_slurm(stage, dependency=last_prep_job)

    train_models_job = None
    output_status, _ = stage_output_status("train_models")
    if output_status != "DONE":
        existing = reusable_job("train_models")
        train_models_job = existing if existing is not None else submit_slurm("train_models", dependency=last_prep_job)

    train_committee_job = None
    output_status, _ = stage_output_status("train_committee")
    if output_status != "DONE":
        existing = reusable_job("train_committee")
        train_committee_job = existing if existing is not None else submit_slurm("train_committee", dependency=last_prep_job)

    eval_job = None
    output_status, _ = stage_output_status("evaluation")
    if output_status != "DONE":
        existing = reusable_job("evaluation")
        eval_job = existing if existing is not None else submit_slurm("evaluation", dependency=train_models_job)

    probe_job = None
    output_status, _ = stage_output_status("fresh_probe")
    if output_status != "DONE":
        existing = reusable_job("fresh_probe")
        probe_job = existing if existing is not None else submit_slurm("fresh_probe", dependency=train_models_job)

    scoring_job = None
    output_status, _ = stage_output_status("probe_scoring")
    if output_status != "DONE":
        existing = reusable_job("probe_scoring")
        if existing is not None:
            scoring_job = existing
        else:
            scoring_deps = [j for j in [probe_job, train_committee_job] if j is not None]
            dep_arg = ",".join(str(j) for j in scoring_deps) if scoring_deps else None
            scoring_job = submit_slurm("probe_scoring", dependency=dep_arg)

    conv_job = None
    output_status, _ = stage_output_status("convergence_report")
    if output_status != "DONE":
        existing = reusable_job("convergence_report")
        if existing is not None:
            conv_job = existing
        else:
            conv_deps = [j for j in [eval_job, scoring_job] if j is not None]
            dep_arg = ",".join(str(j) for j in conv_deps) if conv_deps else None
            conv_job = submit_slurm("convergence_report", dependency=dep_arg)


def run_round():

    print()
    print("=" * 78)

    print(
        f"ACTIVE-LEARNING ROUND "
        f"v{cfg.ROUND}"
    )

    print("=" * 78)

    # ------------------------------------------------------------------
    # v0 Bootstrap Branch (sample_v0 -> dft -> post_dft_chain)
    # ------------------------------------------------------------------
    if cfg.ROUND == 0:
        sample_job = None
        if sample_v0_complete():
            print("sample_v0: already complete (400 structures generated)")
        else:
            sample_job = reusable_job("sample_v0")
            if sample_job is None:
                sample_job = submit_slurm("sample_v0")
                print(f"Submitted sample_v0: {sample_job}")
            else:
                print(f"sample_v0: using active job {sample_job}")

        dft_job = None
        if not dft_complete():
            dft_job = reusable_job("dft")
            if dft_job is None:
                dft_dependency = sample_job
                dft_job = submit_slurm("dft", dependency=dft_dependency)
                print(f"Submitted automated DFT dispatcher: {dft_job} (dependency: {dft_dependency})")
            else:
                print(f"dft: using active dispatcher job {dft_job}")
        else:
            print("dft: calculations already complete")

        launch_post_dft_chain(dependency=dft_job)
        return

    # ------------------------------------------------------------------
    # Committee scoring (Inherited from previous probe or freshly evaluated)
    # ------------------------------------------------------------------

    score_status, _ = stage_output_status("committee_scoring")
    if score_status == "DONE":
        score_job = None
        print("committee_scoring: already complete")
    else:
        score_job = reusable_job("committee_scoring")
        if score_job is None:
            score_job = submit_slurm(
                "committee_scoring",
                dependency=None,
            )
        else:
            print(
                "committee_scoring: using "
                f"job {score_job}"
            )

    # ------------------------------------------------------------------
    # Candidate selection
    # ------------------------------------------------------------------

    if file_ok(
        cfg.SELECTION_MANIFEST
    ):

        selection_job = None

        print(
            "candidate_selection: already complete"
        )

    else:

        selection_job = reusable_job(
            "candidate_selection"
        )

        if selection_job is None:

            selection_dependency = score_job

            selection_job = submit_slurm(
                "candidate_selection",
                dependency=selection_dependency,
            )

        else:

            print(
                "candidate_selection: using "
                f"job {selection_job}"
            )

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # DFT input preparation & Automated Dispatch Chain
    # ------------------------------------------------------------------

    run_prepare_dft()

    dft_job = None
    if not dft_complete():
        dft_job = reusable_job("dft")
        if dft_job is None:
            dft_dependency = selection_job or score_job
            dft_job = submit_slurm("dft", dependency=dft_dependency)
            print(f"Submitted automated DFT dispatcher: {dft_job} (dependency: {dft_dependency})")
        else:
            print(f"dft: using active dispatcher job {dft_job}")
    else:
        print("dft: calculations already complete")

    launch_post_dft_chain(dependency=dft_job)


# ============================================================================
# CLI
# ============================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "One-button MACE-QEq active-learning orchestrator."
        )
    )

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        "--status",
        action="store_true",
        help="Show workflow state.",
    )

    group.add_argument(
        "--run-round",
        action="store_true",
        help="Submit/reuse the complete active-learning round.",
    )

    parser.add_argument(
        "round_pos",
        nargs="?",
        type=int,
        default=None,
        help="Target round index (e.g. 0, 1, 2). Auto-detected if omitted.",
    )

    parser.add_argument(
        "--round",
        dest="round_flag",
        type=int,
        default=None,
        help="Explicit target round index.",
    )

    return parser.parse_args()


def main():

    args = parse_args()
    target_round = args.round_pos if args.round_pos is not None else args.round_flag
    if target_round is None:
        target_round = detect_current_round()

    sync_config_round(target_round)
    cfg.validate_config()
    ensure_directories()
    initialize_state()

    if args.status:

        print_status(
            refresh_state()
        )

        return

    if args.run_round:

        run_round()

        return


if __name__ == "__main__":

    main()
