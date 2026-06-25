"""
comparison.py
=============
Compare baseline EEG features against task EEG features for five cognitive
constructs: flow state, cognitive load, attention/engagement, System 1 vs
System 2 thinking, and valence.

Each compare_* function accepts pre-computed feature dicts (produced by the
existing analysis modules) and returns a structured result dict containing:
  - per-criterion deltas (task - baseline)
  - whether each criterion is satisfied
  - an overall boolean determination
  - a human-readable summary string

Usage example
-------------
    from load_preprocess import load_galea_txt, load_baseline_galea, filter_data, remove_blinks
    from flow_state   import compute_flow_features
    from system1_2    import compute_sys12_features
    from attention    import compute_attn_features, compute_ddtf_connectivity
    from valence      import compute_beta_alpha_ratio
    from comparison   import (
        compare_flow_state, compare_cognitive_load,
        compare_attention, compare_system12, compare_valence,
        print_comparison_report,
    )

    # --- preprocess ---
    baseline_raw = filter_data(load_baseline_galea(BASELINE_FILE, CH_NAMES, SFREQ)[4])
    baseline_raw = remove_blinks(baseline_raw)           # use sw2 as resting baseline
    task_raw     = filter_data(load_galea_txt(TASK_FILE, CH_NAMES, SFREQ))
    task_raw     = remove_blinks(task_raw)

    # --- extract features ---
    bl_flow  = compute_flow_features(baseline_raw)
    bl_sys12 = compute_sys12_features(baseline_raw)
    bl_attn  = compute_attn_features(baseline_raw)
    bl_ddtf  = compute_ddtf_connectivity(baseline_raw)
    bl_val   = compute_beta_alpha_ratio(baseline_raw)

    tk_flow  = compute_flow_features(task_raw)
    tk_sys12 = compute_sys12_features(task_raw)
    tk_attn  = compute_attn_features(task_raw)
    tk_ddtf  = compute_ddtf_connectivity(task_raw)
    tk_val   = compute_beta_alpha_ratio(task_raw)

    # --- compare ---
    results = {
        "flow_state"      : compare_flow_state(bl_flow, tk_flow),
        "cognitive_load"  : compare_cognitive_load(bl_sys12, tk_sys12),
        "attention"       : compare_attention(bl_attn, tk_attn, bl_ddtf, tk_ddtf),
        "system_1_2"      : compare_system12(bl_flow, tk_flow, bl_sys12, tk_sys12),
        "valence"         : compare_valence(bl_val, tk_val),
    }
    print_comparison_report(results)
"""

from __future__ import annotations

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _delta(task_val: float, baseline_val: float) -> float:
    """Signed change: task − baseline."""
    return task_val - baseline_val


def _pct_change(task_val: float, baseline_val: float) -> float:
    """Percentage change relative to baseline (returns NaN if baseline is 0)."""
    if baseline_val == 0:
        return float("nan")
    return 100.0 * (task_val - baseline_val) / abs(baseline_val)


def _criterion(label: str, task_val: float, baseline_val: float,
               direction: str) -> dict:
    """
    Build a single criterion result.

    Parameters
    ----------
    label      : human-readable name for this criterion
    task_val   : scalar from the task recording
    baseline_val : scalar from the baseline recording
    direction  : 'higher' if task should exceed baseline, 'lower' otherwise

    Returns
    -------
    dict with keys: label, baseline, task, delta, pct_change, met
    """
    d = _delta(task_val, baseline_val)
    met = (d > 0) if direction == "higher" else (d < 0)
    return {
        "label"      : label,
        "baseline"   : baseline_val,
        "task"       : task_val,
        "delta"      : d,
        "pct_change" : _pct_change(task_val, baseline_val),
        "direction"  : direction,
        "met"        : met,
    }


def _ddtf_mean(ddtf_result: dict, band: str) -> float:
    """
    Collapse an (n_ch × n_ch) dDTF connectivity matrix to a single scalar
    by averaging all off-diagonal entries (i.e. actual directed connections).
    """
    mat = ddtf_result[band]
    n = mat.shape[0]
    mask = ~np.eye(n, dtype=bool)          # exclude self-connections on diagonal
    return float(mat[mask].mean())


# ─────────────────────────────────────────────────────────────────────────────
# 1. Flow state
# ─────────────────────────────────────────────────────────────────────────────

def compare_flow_state(baseline_flow: dict, task_flow: dict) -> dict:
    """
    Determine whether the task recording shows EEG signatures of flow state
    relative to baseline.

    Criteria (all must be satisfied for a positive determination):
      ✓ frontal theta power increases
      ✓ frontal alpha power increases
      ✓ right central alpha power increases
      ✓ global theta power increases
      ✓ global alpha power increases
      ✓ global beta power decreases

    Parameters
    ----------
    baseline_flow : dict returned by compute_flow_features() for the baseline
    task_flow     : dict returned by compute_flow_features() for the task

    Returns
    -------
    dict with keys:
        criteria  – list of per-criterion result dicts
        detected  – True if all criteria are satisfied
        summary   – human-readable description
    """
    criteria = [
        _criterion("Frontal theta ↑",        task_flow["frontal_theta"],        baseline_flow["frontal_theta"],        "higher"),
        _criterion("Frontal alpha ↑",         task_flow["frontal_alpha"],        baseline_flow["frontal_alpha"],        "higher"),
        _criterion("Right central alpha ↑",   task_flow["right_central_alpha"],  baseline_flow["right_central_alpha"],  "higher"),
        _criterion("Global theta power ↑",    task_flow["theta_power"],          baseline_flow["theta_power"],          "higher"),
        _criterion("Global alpha power ↑",    task_flow["alpha_power"],          baseline_flow["alpha_power"],          "higher"),
        _criterion("Global beta power ↑",     task_flow["beta_power"],           baseline_flow["beta_power"],           "higher"),
    ]

    n_met    = sum(c["met"] for c in criteria)
    detected = n_met == len(criteria)

    summary = (
        f"FLOW STATE {'DETECTED' if detected else 'NOT DETECTED'} "
        f"({n_met}/{len(criteria)} criteria met)"
    )
    return {"criteria": criteria, "detected": detected, "summary": summary}


# ─────────────────────────────────────────────────────────────────────────────
# 2. Cognitive load
# ─────────────────────────────────────────────────────────────────────────────

def compare_cognitive_load(baseline_sys12: dict, task_sys12: dict) -> dict:
    """
    Determine whether the task recording shows elevated cognitive load relative
    to baseline.

    Criteria:
      ✓ Frontal midline theta increases  (higher working-memory load)
      ✓ Parietal alpha decreases         (attentional engagement suppresses parietal alpha)

    Parameters
    ----------
    baseline_sys12 : dict returned by compute_sys12_features() for the baseline
    task_sys12     : dict returned by compute_sys12_features() for the task

    Returns
    -------
    Same structure as compare_flow_state().
    """
    criteria = [
        _criterion("Frontal midline theta ↑", task_sys12["frontal_theta"],  baseline_sys12["frontal_theta"],  "higher"),
        _criterion("Parietal alpha ↓",         task_sys12["parietal_alpha"], baseline_sys12["parietal_alpha"], "lower"),
    ]

    n_met    = sum(c["met"] for c in criteria)
    detected = n_met == len(criteria)

    summary = (
        f"COGNITIVE LOAD {'ELEVATED' if detected else 'NOT ELEVATED'} "
        f"({n_met}/{len(criteria)} criteria met)"
    )
    return {"criteria": criteria, "detected": detected, "summary": summary}


# ─────────────────────────────────────────────────────────────────────────────
# 3. Attention / engagement
# ─────────────────────────────────────────────────────────────────────────────

def compare_attention(baseline_attn: dict, task_attn: dict,
                      baseline_ddtf: dict, task_ddtf: dict) -> dict:
    """
    Determine whether the task recording shows heightened attention/engagement
    relative to baseline.

    Criteria:
      ✓ Global beta power increases
      ✓ Global alpha power decreases
      ✓ Mean alpha-band dDTF connectivity increases
      ✓ Mean beta-band dDTF connectivity increases
      ✓ Mean delta-band dDTF connectivity increases

    Parameters
    ----------
    baseline_attn : dict returned by compute_attn_features() for the baseline
    task_attn     : dict returned by compute_attn_features() for the task
    baseline_ddtf : dDTFResult returned by compute_ddtf_connectivity() for baseline
    task_ddtf     : dDTFResult returned by compute_ddtf_connectivity() for task

    Returns
    -------
    Same structure as compare_flow_state().
    """
    bl_alpha_conn = _ddtf_mean(baseline_ddtf, "alpha")
    tk_alpha_conn = _ddtf_mean(task_ddtf,     "alpha")
    bl_beta_conn  = _ddtf_mean(baseline_ddtf, "beta")
    tk_beta_conn  = _ddtf_mean(task_ddtf,     "beta")
    bl_delta_conn = _ddtf_mean(baseline_ddtf, "delta")
    tk_delta_conn = _ddtf_mean(task_ddtf,     "delta")

    criteria = [
        _criterion("Global beta power ↑",          task_attn["beta_power"],  baseline_attn["beta_power"],  "higher"),
        _criterion("Global alpha power ↓",          task_attn["alpha_power"], baseline_attn["alpha_power"], "lower"),
        _criterion("Alpha-band connectivity ↑",     tk_alpha_conn,            bl_alpha_conn,                "higher"),
        _criterion("Beta-band connectivity ↑",      tk_beta_conn,             bl_beta_conn,                 "higher"),
        _criterion("Delta-band connectivity ↑",     tk_delta_conn,            bl_delta_conn,                "higher"),
    ]

    n_met    = sum(c["met"] for c in criteria)
    detected = n_met == len(criteria)

    summary = (
        f"ATTENTION/ENGAGEMENT {'ELEVATED' if detected else 'NOT ELEVATED'} "
        f"({n_met}/{len(criteria)} criteria met)"
    )
    return {"criteria": criteria, "detected": detected, "summary": summary}


# ─────────────────────────────────────────────────────────────────────────────
# 4. System 1 / System 2 thinking
# ─────────────────────────────────────────────────────────────────────────────

def compare_system12(baseline_flow: dict, task_flow: dict,
                     baseline_sys12: dict, task_sys12: dict) -> dict:
    """
    Classify the task recording as predominantly System 1 or System 2 thinking.

    System 1 (fast, intuitive):
      ✓ Frontal theta decreases relative to baseline

    System 2 (slow, deliberate):
      ✓ Frontal theta increases relative to baseline
      ✓ Global beta power increases relative to baseline

    Note: the two systems are mutually exclusive. System 2 criteria take
    precedence — if both frontal theta and beta are elevated, System 2 is
    reported. If frontal theta is reduced (and beta is not elevated), System 1
    is reported. Mixed or ambiguous patterns are flagged accordingly.

    Parameters
    ----------
    baseline_flow  : dict from compute_flow_features() for baseline
    task_flow      : dict from compute_flow_features() for task
    baseline_sys12 : dict from compute_sys12_features() for baseline
    task_sys12     : dict from compute_sys12_features() for task

    Returns
    -------
    dict with keys:
        criteria_sys1  – list of per-criterion dicts for System 1
        criteria_sys2  – list of per-criterion dicts for System 2
        system         – 'System 1', 'System 2', or 'Ambiguous'
        summary        – human-readable description
    """
    # System 1: frontal theta decreases
    crit_sys1 = [
        _criterion("Frontal theta ↓ (System 1)", task_sys12["frontal_theta"], baseline_sys12["frontal_theta"], "lower"),
    ]

    # System 2: frontal theta increases AND beta increases
    crit_sys2 = [
        _criterion("Frontal theta ↑ (System 2)", task_sys12["frontal_theta"], baseline_sys12["frontal_theta"], "higher"),
        _criterion("Global beta power ↑ (System 2)", task_flow["beta_power"], baseline_flow["beta_power"],     "higher"),
    ]

    sys2_met = all(c["met"] for c in crit_sys2)
    sys1_met = all(c["met"] for c in crit_sys1) and not sys2_met

    if sys2_met:
        system  = "System 2"
        summary = "SYSTEM 2 (deliberate/analytical) thinking detected"
    elif sys1_met:
        system  = "System 1"
        summary = "SYSTEM 1 (fast/intuitive) thinking detected"
    else:
        system  = "Ambiguous"
        summary = "AMBIGUOUS — pattern does not clearly match System 1 or System 2"

    n_sys1 = sum(c["met"] for c in crit_sys1)
    n_sys2 = sum(c["met"] for c in crit_sys2)
    summary += (
        f" | Sys1 criteria: {n_sys1}/{len(crit_sys1)}, "
        f"Sys2 criteria: {n_sys2}/{len(crit_sys2)}"
    )

    return {
        "criteria_sys1" : crit_sys1,
        "criteria_sys2" : crit_sys2,
        "system"        : system,
        "summary"       : summary,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. Valence
# ─────────────────────────────────────────────────────────────────────────────

def compare_valence(baseline_valence: dict, task_valence: dict) -> dict:
    """
    Assess valence shift between baseline and task recordings.

    A higher beta/alpha ratio during the task indicates more positive or
    aroused valence (increased approach motivation).

    Criteria:
      ✓ Mean beta/alpha ratio increases

    Parameters
    ----------
    baseline_valence : BetaAlphaResult from compute_beta_alpha_ratio() for baseline
    task_valence     : BetaAlphaResult from compute_beta_alpha_ratio() for task

    Returns
    -------
    Same structure as compare_flow_state(), plus:
        baseline_mean_ratio – scalar
        task_mean_ratio     – scalar
        interpretation      – 'More positive/aroused', 'More negative/withdrawn', or 'Neutral'
    """
    bl_ratio = baseline_valence["mean_ratio"]
    tk_ratio = task_valence["mean_ratio"]

    criteria = [
        _criterion("Beta/alpha ratio ↑", tk_ratio, bl_ratio, "higher"),
    ]

    n_met    = sum(c["met"] for c in criteria)
    detected = n_met == len(criteria)

    if detected:
        interpretation = "More positive / aroused valence"
    else:
        interpretation = "More negative / withdrawn valence"

    summary = (
        f"VALENCE: {interpretation} "
        f"(baseline β/α = {bl_ratio:.3f}, task β/α = {tk_ratio:.3f}, "
        f"Δ = {tk_ratio - bl_ratio:+.3f})"
    )

    return {
        "criteria"            : criteria,
        "detected"            : detected,
        "baseline_mean_ratio" : bl_ratio,
        "task_mean_ratio"     : tk_ratio,
        "interpretation"      : interpretation,
        "summary"             : summary,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Report printer
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_criterion(c: dict, indent: str = "    ") -> str:
    """Format a single criterion as a readable line."""
    tick  = "✓" if c["met"] else "✗"
    arrow = "↑" if c["direction"] == "higher" else "↓"
    pct   = c["pct_change"]
    pct_str = f"{pct:+.1f}%" if not np.isnan(pct) else "N/A"
    return (
        f"{indent}{tick} {c['label']:<35s} "
        f"baseline={c['baseline']:.4g}  task={c['task']:.4g}  "
        f"Δ={c['delta']:+.4g}  ({pct_str})"
    )


def print_comparison_report(results: dict) -> None:
    """
    Print a structured, human-readable comparison report for all five
    cognitive constructs.

    Parameters
    ----------
    results : dict mapping construct name → result dict, e.g.:
        {
            "flow_state"     : compare_flow_state(...),
            "cognitive_load" : compare_cognitive_load(...),
            "attention"      : compare_attention(...),
            "system_1_2"     : compare_system12(...),
            "valence"        : compare_valence(...),
        }
    """
    SEPARATOR = "=" * 72

    print(f"\n{SEPARATOR}")
    print("  EEG COGNITIVE STATE COMPARISON REPORT")
    print(SEPARATOR)

    # ── Flow state ────────────────────────────────────────────────────────
    if "flow_state" in results:
        r = results["flow_state"]
        print(f"\n{'─'*72}")
        print(f"  1. FLOW STATE")
        print(f"{'─'*72}")
        print(f"  {r['summary']}")
        for c in r["criteria"]:
            print(_fmt_criterion(c))

    # ── Cognitive load ────────────────────────────────────────────────────
    if "cognitive_load" in results:
        r = results["cognitive_load"]
        print(f"\n{'─'*72}")
        print(f"  2. COGNITIVE LOAD")
        print(f"{'─'*72}")
        print(f"  {r['summary']}")
        for c in r["criteria"]:
            print(_fmt_criterion(c))

    # ── Attention / engagement ────────────────────────────────────────────
    if "attention" in results:
        r = results["attention"]
        print(f"\n{'─'*72}")
        print(f"  3. ATTENTION / ENGAGEMENT")
        print(f"{'─'*72}")
        print(f"  {r['summary']}")
        for c in r["criteria"]:
            print(_fmt_criterion(c))

    # ── System 1 / System 2 ───────────────────────────────────────────────
    if "system_1_2" in results:
        r = results["system_1_2"]
        print(f"\n{'─'*72}")
        print(f"  4. SYSTEM 1 / SYSTEM 2")
        print(f"{'─'*72}")
        print(f"  {r['summary']}")
        print("  System 1 criteria:")
        for c in r["criteria_sys1"]:
            print(_fmt_criterion(c, indent="      "))
        print("  System 2 criteria:")
        for c in r["criteria_sys2"]:
            print(_fmt_criterion(c, indent="      "))

    # ── Valence ───────────────────────────────────────────────────────────
    if "valence" in results:
        r = results["valence"]
        print(f"\n{'─'*72}")
        print(f"  5. VALENCE")
        print(f"{'─'*72}")
        print(f"  {r['summary']}")
        for c in r["criteria"]:
            print(_fmt_criterion(c))

    print(f"\n{SEPARATOR}\n")