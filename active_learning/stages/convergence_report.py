#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
AL_DIR = SCRIPT_DIR.parent
if str(AL_DIR) not in sys.path:
    sys.path.insert(0, str(AL_DIR))

import al_config as cfg

PERCENTILE_DIST_P95 = 95.0

def require_file(path: Path) -> None:
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing file: {path}")

def safe_pct_change(old: float, new: float) -> float:
    return float("nan") if old == 0.0 else (new - old) / old

def bool_text(value: bool) -> str:
    return "PASS" if value else "FAIL"

def get_summary_row(df: pd.DataFrame, generation: int, dataset: str, system: str = "ALL") -> pd.Series:
    rows = df[(df["generation"] == f"v{generation}") & (df["dataset"] == dataset) & (df["system"] == system)]
    if len(rows) != 1:
        raise RuntimeError(f"Expected one row for v{generation} {dataset} {system}, found {len(rows)}")
    return rows.iloc[0]

def analyze_evaluation(summary: pd.DataFrame) -> dict:
    current_test = get_summary_row(summary, cfg.ROUND, "independent_test", "ALL")
    f_current_full = float(current_test["F_RMSE_full_eV_A"])
    f_current_short = float(current_test["F_RMSE_shortQEq_eV_A"])

    if cfg.PARENT_ROUND < 0:
        # v0 baseline: benchmark kiem dinh doc lap tuyet doi theo Hu et al. 2025 (F_RMSE <= 0.163 eV/A tren ConstQ)
        test_guardrail_pass = f_current_short <= 0.180
        return {
            "independent_test": {
                "frames": int(current_test["frames"]),
                "current_full_E_RMSE_meV_atom": float(current_test["E_RMSE_full_meV_atom"]),
                "current_shortQEq_E_RMSE_meV_atom": float(current_test["E_RMSE_shortQEq_meV_atom"]),
                "current_full_F_RMSE_eV_A": f_current_full,
                "current_shortQEq_F_RMSE_eV_A": f_current_short,
                "full_force_change_fraction": 0.0,
                "shortQEq_force_change_fraction": 0.0,
                "force_guardrail_pass": bool(test_guardrail_pass),
            },
            "current_enrichment": {
                "dataset": "bootstrap_v0",
                "full_force_change_fraction": 0.0,
                "shortQEq_force_change_fraction": 0.0,
                "learning_pass": True,
            },
            "historical_holdouts": [],
            "historical_holdout_guardrail_pass": True,
        }

    parent_test = get_summary_row(summary, cfg.PARENT_ROUND, "independent_test", "ALL")
    f_parent_full = float(parent_test["F_RMSE_full_eV_A"])
    f_parent_short = float(parent_test["F_RMSE_shortQEq_eV_A"])

    full_reg = safe_pct_change(f_parent_full, f_current_full)
    short_reg = safe_pct_change(f_parent_short, f_current_short)
    test_guardrail_pass = (
        full_reg <= cfg.CONV_MAX_TEST_FORCE_RMSE_REGRESSION and short_reg <= cfg.CONV_MAX_TEST_FORCE_RMSE_REGRESSION
    )

    enrichment_name = f"v{cfg.ROUND}_enrichment"
    p_enrich = get_summary_row(summary, cfg.PARENT_ROUND, enrichment_name, "ALL")
    c_enrich = get_summary_row(summary, cfg.ROUND, enrichment_name, "ALL")
    enrich_full_change = safe_pct_change(float(p_enrich["F_RMSE_full_eV_A"]), float(c_enrich["F_RMSE_full_eV_A"]))
    enrich_short_change = safe_pct_change(float(p_enrich["F_RMSE_shortQEq_eV_A"]), float(c_enrich["F_RMSE_shortQEq_eV_A"]))
    enrichment_pass = enrich_short_change < 0.0

    holdout_results = []
    catastrophic_holdout_fail = False
    holdouts = [f"v{generation}_holdout" for generation in range(1, cfg.ROUND + 1)]

    for d in holdouts:
        p_rows = summary[(summary["generation"] == f"v{cfg.PARENT_ROUND}") & (summary["dataset"] == d) & (summary["system"] == "ALL")]
        c_rows = summary[(summary["generation"] == f"v{cfg.ROUND}") & (summary["dataset"] == d) & (summary["system"] == "ALL")]
        if len(p_rows) != 1 or len(c_rows) != 1:
            raise RuntimeError(
                f"{d}: expected exactly one parent and current ALL-system evaluation; "
                f"found parent={len(p_rows)}, current={len(c_rows)}."
            )
        p = p_rows.iloc[0]
        c = c_rows.iloc[0]
        f_chg = safe_pct_change(float(p["F_RMSE_full_eV_A"]), float(c["F_RMSE_full_eV_A"]))
        s_chg = safe_pct_change(float(p["F_RMSE_shortQEq_eV_A"]), float(c["F_RMSE_shortQEq_eV_A"]))
        h_pass = f_chg <= cfg.CONV_MAX_HOLDOUT_FORCE_RMSE_REGRESSION and s_chg <= cfg.CONV_MAX_HOLDOUT_FORCE_RMSE_REGRESSION
        if not h_pass:
            catastrophic_holdout_fail = True
        holdout_results.append({
            "dataset": d,
            "full_force_change_fraction": f_chg,
            "shortQEq_force_change_fraction": s_chg,
            "pass": bool(h_pass),
        })

    return {
        "independent_test": {
            "frames": int(current_test["frames"]),
            "parent_full_E_RMSE_meV_atom": float(parent_test["E_RMSE_full_meV_atom"]),
            "current_full_E_RMSE_meV_atom": float(current_test["E_RMSE_full_meV_atom"]),
            "parent_shortQEq_E_RMSE_meV_atom": float(parent_test["E_RMSE_shortQEq_meV_atom"]),
            "current_shortQEq_E_RMSE_meV_atom": float(current_test["E_RMSE_shortQEq_meV_atom"]),
            "parent_full_F_RMSE_eV_A": f_parent_full,
            "current_full_F_RMSE_eV_A": f_current_full,
            "parent_shortQEq_F_RMSE_eV_A": f_parent_short,
            "current_shortQEq_F_RMSE_eV_A": f_current_short,
            "full_force_change_fraction": full_reg,
            "shortQEq_force_change_fraction": short_reg,
            "force_guardrail_pass": bool(test_guardrail_pass),
        },
        "current_enrichment": {
            "dataset": enrichment_name,
            "full_force_change_fraction": enrich_full_change,
            "shortQEq_force_change_fraction": enrich_short_change,
            "learning_pass": bool(enrichment_pass),
        },
        "historical_holdouts": holdout_results,
        "historical_holdout_guardrail_pass": bool(not catastrophic_holdout_fail),
    }

def analyze_probe(scores: pd.DataFrame) -> dict:
    u = pd.to_numeric(scores["sigma_F_p95_eV_A"], errors="coerce").to_numpy(dtype=np.float64)
    if abs(len(u) - cfg.PROBE_EXPECTED_MD_FRAMES) > 2 or not np.all(np.isfinite(u)):
        raise RuntimeError("Fresh probe scores corrupted or incomplete.")
    f_target = float(np.mean(u >= cfg.CONV_SIGMA_P95_TARGET))
    f_hard = float(np.mean(u >= cfg.CONV_SIGMA_P95_HARD))
    t_pass = f_target <= cfg.CONV_MAX_FRAC_EXCEED_TARGET
    h_pass = f_hard <= cfg.CONV_MAX_FRAC_EXCEED_HARD
    return {
        "frames": int(len(u)),
        "sigma_p95_mean_eV_A": float(np.mean(u)),
        "sigma_p95_median_eV_A": float(np.median(u)),
        "sigma_p95_distribution_p95_eV_A": float(np.percentile(u, PERCENTILE_DIST_P95)),
        "sigma_p95_max_eV_A": float(np.max(u)),
        "fraction_ge_target": f_target,
        "fraction_ge_hard": f_hard,
        "target_pass": bool(t_pass),
        "hard_pass": bool(h_pass),
        "probe_pass": bool(t_pass and h_pass),
    }

def decide_recommendation(evaluation: dict, probe: dict) -> tuple[str, list[str]]:
    reasons = []
    test_pass = evaluation["independent_test"]["force_guardrail_pass"]
    holdout_pass = evaluation["historical_holdout_guardrail_pass"]
    enrich_pass = evaluation["current_enrichment"]["learning_pass"]
    probe_pass = probe["probe_pass"]

    if not probe_pass:
        reasons.append("Fresh probe exceeds uncertainty limit.")
        return "CONTINUE_AL", reasons
    if cfg.PARENT_ROUND < 0 and test_pass:
        reasons.append("Bootstrap v0 baseline verified. Proceeding to active learning round v1.")
        return "CONTINUE_AL", reasons
    if test_pass and holdout_pass and enrich_pass:
        reasons.append("All convergence and quality criteria satisfied.")
        return "CONVERGED", reasons
    if not test_pass:
        reasons.append("Independent test force RMSE exceeded regression guardrail.")
    if not holdout_pass:
        reasons.append("Historical holdout force RMSE exceeded regression guardrail.")
    if not enrich_pass:
        reasons.append("Short+QEq did not improve on enrichment set.")
    return "REVIEW", reasons

def write_historical_errors(current_summary: pd.DataFrame) -> tuple[Path, Path]:
    summary_rows = []
    element_rows = []
    for generation in range(cfg.ROUND + 1):
        evaluation_dir = cfg.MD_ROOT / f"active_learning_v{generation}" / "validation" / "evaluation"
        prefix = f"v{generation}_standalone" if generation == 0 else f"v{generation - 1}_vs_v{generation}"
        for suffix, output in (("summary", summary_rows), ("elements", element_rows)):
            path = evaluation_dir / f"{prefix}_{suffix}.csv"
            if generation == cfg.ROUND and suffix == "summary":
                df = current_summary
            elif path.is_file() and path.stat().st_size > 0:
                df = pd.read_csv(path)
            else:
                continue
            required = {"generation", "dataset"}
            if not required.issubset(df.columns):
                raise RuntimeError(f"Historical evaluation columns missing: {path}")
            selected = df[(df["generation"] == f"v{generation}") & (df["dataset"] == "independent_test")].copy()
            if not selected.empty:
                output.append(selected)

    def save_history(rows, suffix, columns):
        if rows:
            df = pd.concat(rows, ignore_index=True)
            keys = ["generation", "dataset"] + (["system", "element"] if suffix == "elements" and "system" in df else ["system"] if suffix == "summary" else ["element"])
            if df.duplicated(keys).any():
                raise RuntimeError(f"Duplicate historical {suffix} rows")
        else:
            df = pd.DataFrame(columns=columns)
        df["round_number"] = pd.to_numeric(df["generation"].str.removeprefix("v"), errors="raise") if not df.empty else pd.Series(dtype=int)
        df = df.sort_values(["round_number"] + (["system", "element"] if suffix == "elements" and "system" in df else ["system"] if suffix == "summary" else ["element"] if suffix == "elements" else []))
        df = df.drop(columns="round_number")
        path = cfg.CONVERGENCE_DIR / f"historical_{suffix}.csv"
        df.to_csv(path, index=False)
        return path

    summary_path = save_history(summary_rows, "summary", ["generation", "dataset", "system", "frames"])
    elements_path = save_history(element_rows, "elements", ["generation", "dataset", "element", "atoms"])
    return summary_path, elements_path


def main() -> None:
    cfg.validate_config()
    require_file(cfg.EVALUATION_SUMMARY)
    require_file(cfg.PROBE_SCORE_FILE)
    summary = pd.read_csv(cfg.EVALUATION_SUMMARY)
    probe_scores = pd.read_csv(cfg.PROBE_SCORE_FILE)
    ev = analyze_evaluation(summary)
    pr = analyze_probe(probe_scores)
    rec, rz = decide_recommendation(ev, pr)

    cfg.CONVERGENCE_DIR.mkdir(parents=True, exist_ok=True)
    historical_summary, historical_elements = write_historical_errors(summary)
    with cfg.CONVERGENCE_JSON.open("w", encoding="utf-8") as handle:
        json.dump({
            "round": cfg.ROUND,
            "parent_round": cfg.PARENT_ROUND,
            "system": cfg.SYSTEM,
            "evaluation": ev,
            "fresh_probe": pr,
            "recommendation": rec,
            "reasons": rz,
        }, handle, indent=2, sort_keys=True)

    lines = [
        f"AL-v{cfg.ROUND} CONVERGENCE REPORT ({cfg.SYSTEM})",
        f"RECOMMENDATION: {rec}",
        f"Fresh probe pass: {bool_text(pr['probe_pass'])}",
        f"Independent test guardrail: {bool_text(ev['independent_test']['force_guardrail_pass'])}",
        f"Holdout guardrail: {bool_text(ev['historical_holdout_guardrail_pass'])}",
        f"Enrichment learning: {bool_text(ev['current_enrichment']['learning_pass'])}",
        f"Historical E/F by system: {historical_summary}",
        f"Historical force errors by element: {historical_elements}",
    ]
    cfg.CONVERGENCE_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"REPORT SAVED: {cfg.CONVERGENCE_JSON}")

if __name__ == "__main__":
    main()
