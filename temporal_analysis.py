"""
temporal_analysis.py
====================
Detect when each cognitive state is first reached during a task recording,
then generate dDTF network diagrams (all bands) at those moments.

Sliding 2-second windows (matching valence.py) are compared against fixed
baseline features using the same criteria as comparison.py.  dDTF snapshots
use a longer centred window (default 30 s) because MVAR needs sufficient data.
"""

from __future__ import annotations

import os
from typing import TypedDict

import mne
import numpy as np

from attention import (
    EEG_CHANNELS,
    ALPHA_BAND,
    BETA_BAND,
    compute_ddtf_connectivity,
    plot_all_bands,
    plot_ddtf,
)
from comparison import (
    compare_cognitive_load,
    compare_flow_state,
    compare_system12,
    _criterion,
)
from flow_state import (
    FRONTAL_CHS,
    RT_CENTRAL_CH,
    THETA_BAND,
    ALPHA_BAND as FLOW_ALPHA_BAND,
    BETA_BAND as FLOW_BETA_BAND,
)
from system1_2 import PARIETAL_CHS, FRONTAL_CHS as SYS12_FRONTAL_CHS
from valence import WINDOW_SEC, OVERLAP_SEC

# Minimum segment length for stable MVAR fitting (seconds)
DDTF_WINDOW_SEC = 30.0


class ConsecutiveStreak(TypedDict):
    start_sec: float
    end_sec: float
    duration_sec: float
    n_windows: int


class StateOnsetResult(TypedDict):
    state: str
    first_onset_sec: float | None
    longest_streak: ConsecutiveStreak | None
    all_onset_times: list[float]
    n_windows_active: int
    n_windows_total: int
    summary: str


class StateMasks(TypedDict):
    flow_state: np.ndarray
    cognitive_load: np.ndarray
    attention: np.ndarray
    system_1: np.ndarray
    system_2: np.ndarray
    valence: np.ndarray


class TemporalAnalysisResult(TypedDict):
    window_sec: float
    ddtf_window_sec: float
    times: np.ndarray
    masks: StateMasks
    onsets: dict[str, StateOnsetResult]
    ddtf_paths: dict[str, str | None]
    timeline_path: str | None


# ── Shared windowing ──────────────────────────────────────────────────────────

def _window_starts(n_times: int, sfreq: float,
                   window_sec: float, overlap_sec: float) -> np.ndarray:
    win_samp  = int(window_sec * sfreq)
    step_samp = int((window_sec - overlap_sec) * sfreq)
    return np.arange(0, n_times - win_samp + 1, step_samp)


def _welch_psd(epoch: np.ndarray, sfreq: float) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD for a single-channel epoch; returns (psd, freqs)."""
    if epoch.ndim == 1:
        epoch = epoch[np.newaxis, :]
    n_fft = epoch.shape[1]
    spectrum = mne.time_frequency.psd_array_welch(
        epoch,
        sfreq=sfreq,
        fmin=0.5,
        fmax=40.0,
        n_fft=n_fft,
        verbose=False,
    )
    return spectrum[0][0], spectrum[1]


def _band_mean(psd: np.ndarray, freqs: np.ndarray,
               fmin: float, fmax: float) -> float:
    mask = (freqs >= fmin) & (freqs < fmax)
    return float(psd[mask].mean())


def _ch_indices(ch_names: list[str], names: list[str]) -> list[int]:
    return [ch_names.index(ch) for ch in names if ch in ch_names]


# ── Per-window feature extraction ─────────────────────────────────────────────

def _flow_features_from_epoch(data: np.ndarray, ch_names: list[str],
                              sfreq: float) -> dict[str, float]:
    """Compute flow features for one multi-channel epoch."""
    n_ch = len(ch_names)
    theta_pw = np.zeros(n_ch)
    alpha_pw = np.zeros(n_ch)
    beta_pw  = np.zeros(n_ch)

    for ci in range(n_ch):
        psd, freqs = _welch_psd(data[ci], sfreq)
        theta_pw[ci] = _band_mean(psd, freqs, *THETA_BAND)
        alpha_pw[ci] = _band_mean(psd, freqs, *FLOW_ALPHA_BAND)
        beta_pw[ci]  = _band_mean(psd, freqs, *FLOW_BETA_BAND)

    f_idx  = _ch_indices(ch_names, FRONTAL_CHS)
    rc_idx = _ch_indices(ch_names, RT_CENTRAL_CH)

    scale = 1e12
    return {
        "frontal_theta":       float(theta_pw[f_idx].mean()) * scale,
        "frontal_alpha":       float(alpha_pw[f_idx].mean()) * scale,
        "right_central_alpha": float(alpha_pw[rc_idx].mean()) * scale,
        "theta_power":         float(theta_pw.mean()) * scale,
        "alpha_power":         float(alpha_pw.mean()) * scale,
        "beta_power":          float(beta_pw.mean()) * scale,
    }


def _sys12_features_from_epoch(data: np.ndarray, ch_names: list[str],
                               sfreq: float) -> dict[str, float]:
    """Compute cognitive-load features for one multi-channel epoch."""
    target = list(dict.fromkeys(PARIETAL_CHS + SYS12_FRONTAL_CHS))
    idx    = _ch_indices(ch_names, target)
    sub    = data[idx]
    sub_names = [ch_names[i] for i in idx]

    theta_vals, alpha_vals = [], []
    for ci, ch in enumerate(sub_names):
        psd, freqs = _welch_psd(sub[ci], sfreq)
        if ch in PARIETAL_CHS:
            alpha_vals.append(_band_mean(psd, freqs, *FLOW_ALPHA_BAND))
        if ch in SYS12_FRONTAL_CHS:
            theta_vals.append(_band_mean(psd, freqs, *THETA_BAND))

    return {
        "parietal_alpha": float(np.mean(alpha_vals)),
        "frontal_theta":  float(np.mean(theta_vals)),
    }


def _attn_features_from_epoch(data: np.ndarray, ch_names: list[str],
                              sfreq: float) -> dict[str, float]:
    """Compute attention band-power features for one multi-channel epoch."""
    n_ch = len(ch_names)
    alpha_pw = np.zeros(n_ch)
    beta_pw  = np.zeros(n_ch)

    for ci in range(n_ch):
        psd, freqs = _welch_psd(data[ci], sfreq)
        alpha_pw[ci] = _band_mean(psd, freqs, *ALPHA_BAND)
        beta_pw[ci]  = _band_mean(psd, freqs, *BETA_BAND)

    return {
        "alpha_power": float(alpha_pw.mean()),
        "beta_power":  float(beta_pw.mean()),
    }


def _compare_attention_power(baseline_attn: dict, task_attn: dict) -> dict:
    """Attention comparison using band power only (for sliding-window detection)."""
    criteria = [
        _criterion("Global beta power ↑",  task_attn["beta_power"],  baseline_attn["beta_power"],  "higher"),
        _criterion("Global alpha power ↓", task_attn["alpha_power"], baseline_attn["alpha_power"], "lower"),
    ]
    n_met    = sum(c["met"] for c in criteria)
    detected = n_met == len(criteria)
    summary  = (
        f"ATTENTION POWER {'ELEVATED' if detected else 'NOT ELEVATED'} "
        f"({n_met}/{len(criteria)} criteria met)"
    )
    return {"criteria": criteria, "detected": detected, "summary": summary}


def _compare_valence_window(baseline_mean: float, window_ratio: float) -> dict:
    """Valence comparison for a single window ratio vs baseline mean."""
    criteria = [
        _criterion("Beta/alpha ratio ↑", window_ratio, baseline_mean, "higher"),
    ]
    n_met    = sum(c["met"] for c in criteria)
    detected = n_met == len(criteria)
    return {"criteria": criteria, "detected": detected, "summary": ""}


def _system_at_window(baseline_flow: dict, window_flow: dict,
                      baseline_sys12: dict, window_sys12: dict) -> str:
    """Return 'System 1', 'System 2', or '' for ambiguous at one window."""
    result = compare_system12(baseline_flow, window_flow,
                              baseline_sys12, window_sys12)
    system = result["system"]
    return system if system in ("System 1", "System 2") else ""


# ── Onset detection ───────────────────────────────────────────────────────────

def _longest_consecutive_streak(active_mask: np.ndarray) -> tuple[int, int, int]:
    """Return (start_idx, end_idx, length) of the longest True run."""
    best_start, best_end, best_len = -1, -1, 0
    i, n = 0, len(active_mask)

    while i < n:
        if not active_mask[i]:
            i += 1
            continue
        j = i
        while j < n and active_mask[j]:
            j += 1
        run_len = j - i
        if run_len > best_len:
            best_len = run_len
            best_start, best_end = i, j - 1
        i = j

    return best_start, best_end, best_len


def _streak_timestamps(
    times: np.ndarray,
    start_idx: int,
    end_idx: int,
    window_sec: float,
) -> ConsecutiveStreak:
    """Convert a window-index streak to wall-clock start/end/duration."""
    half = window_sec / 2.0
    start_sec = float(times[start_idx] - half)
    end_sec   = float(times[end_idx] + half)
    return ConsecutiveStreak(
        start_sec=start_sec,
        end_sec=end_sec,
        duration_sec=end_sec - start_sec,
        n_windows=end_idx - start_idx + 1,
    )


def _collect_onsets(
    times: np.ndarray,
    active_mask: np.ndarray,
    state_label: str,
    window_sec: float,
) -> StateOnsetResult:
    active_times = times[active_mask].tolist()
    first        = active_times[0] if active_times else None
    n_active     = int(active_mask.sum())
    n_total      = len(times)

    streak_start, streak_end, streak_len = _longest_consecutive_streak(active_mask)
    longest_streak = (
        _streak_timestamps(times, streak_start, streak_end, window_sec)
        if streak_len > 0 else None
    )

    if first is not None:
        pct = 100.0 * n_active / n_total
        summary = (
            f"{state_label}: first reached at {first:.1f} s; "
            f"longest consecutive {longest_streak['duration_sec']:.1f} s "
            f"({longest_streak['start_sec']:.1f}–{longest_streak['end_sec']:.1f} s) "
            f"({n_active}/{n_total} windows active, {pct:.0f}% of task)"
        )
    else:
        summary = f"{state_label}: never reached during task"

    return StateOnsetResult(
        state=state_label,
        first_onset_sec=first,
        longest_streak=longest_streak,
        all_onset_times=active_times,
        n_windows_active=n_active,
        n_windows_total=n_total,
        summary=summary,
    )


def detect_state_onsets(
    task_raw: mne.io.RawArray,
    baseline_flow: dict,
    baseline_sys12: dict,
    baseline_attn: dict,
    baseline_val: dict,
    task_val: dict,
    window_sec: float = WINDOW_SEC,
    overlap_sec: float = OVERLAP_SEC,
) -> tuple[np.ndarray, dict[str, StateOnsetResult], StateMasks]:
    """
    Slide windows across the task and detect when each cognitive state is active.

    Returns centre times for each window and per-state onset results.
    """
    raw_eeg  = task_raw.copy().pick_channels(EEG_CHANNELS)
    data     = raw_eeg.get_data()
    sfreq    = raw_eeg.info["sfreq"]
    ch_names = raw_eeg.ch_names
    n_times  = data.shape[1]

    win_samp  = int(window_sec * sfreq)
    starts    = _window_starts(n_times, sfreq, window_sec, overlap_sec)
    n_win     = len(starts)
    times     = np.zeros(n_win)

    flow_active  = np.zeros(n_win, dtype=bool)
    load_active  = np.zeros(n_win, dtype=bool)
    attn_active  = np.zeros(n_win, dtype=bool)
    sys1_active  = np.zeros(n_win, dtype=bool)
    sys2_active  = np.zeros(n_win, dtype=bool)
    valence_active = np.zeros(n_win, dtype=bool)

    bl_val_mean = baseline_val["mean_ratio"]
    val_times   = task_val["times"]
    val_ratios  = task_val["ratio_per_window"]

    for wi, start in enumerate(starts):
        epoch = data[:, start : start + win_samp]
        times[wi] = (start + win_samp / 2) / sfreq

        w_flow  = _flow_features_from_epoch(epoch, ch_names, sfreq)
        w_sys12 = _sys12_features_from_epoch(epoch, ch_names, sfreq)
        w_attn  = _attn_features_from_epoch(epoch, ch_names, sfreq)

        flow_active[wi]  = compare_flow_state(baseline_flow, w_flow)["detected"]
        load_active[wi]  = compare_cognitive_load(baseline_sys12, w_sys12)["detected"]
        attn_active[wi]  = _compare_attention_power(baseline_attn, w_attn)["detected"]

        system = _system_at_window(baseline_flow, w_flow, baseline_sys12, w_sys12)
        if system == "System 1":
            sys1_active[wi] = True
        elif system == "System 2":
            sys2_active[wi] = True

        # Match valence window by nearest centre time
        vi = int(np.argmin(np.abs(val_times - times[wi])))
        valence_active[wi] = _compare_valence_window(
            bl_val_mean, val_ratios[vi]
        )["detected"]

    masks = StateMasks(
        flow_state=flow_active,
        cognitive_load=load_active,
        attention=attn_active,
        system_1=sys1_active,
        system_2=sys2_active,
        valence=valence_active,
    )
    onsets = {
        "flow_state":     _collect_onsets(times, flow_active,    "Flow state",                window_sec),
        "cognitive_load": _collect_onsets(times, load_active,    "Cognitive load",            window_sec),
        "attention":      _collect_onsets(times, attn_active,    "Attention/engagement",      window_sec),
        "system_1":       _collect_onsets(times, sys1_active,    "System 1 thinking",         window_sec),
        "system_2":       _collect_onsets(times, sys2_active,    "System 2 thinking",         window_sec),
        "valence":        _collect_onsets(times, valence_active, "Positive/aroused valence",  window_sec),
    }
    return times, onsets, masks


# ── State timeline plot ───────────────────────────────────────────────────────

# Top-to-bottom panel order (flow state is topmost).
TIMELINE_PANELS: list[tuple[str, str, str, str]] = [
    ("flow_state",     "Flow state",     "Flow",       "Not flow"),
    ("cognitive_load", "Cognitive load", "High load",  "Not elevated"),
    ("attention",      "Attention",      "Engaged",    "Not engaged"),
    ("system",         "System 1 / 2",   "System 2",   "System 1"),
    ("valence",        "Valence",        "Positive",   "Not positive"),
]


def _system_timeline_values(masks: StateMasks) -> np.ndarray:
    """1 = System 2, 0 = System 1; ambiguous windows are NaN."""
    values = np.full(len(masks["system_1"]), np.nan)
    values[masks["system_2"]] = 1.0
    values[masks["system_1"] & ~masks["system_2"]] = 0.0
    return values


def plot_state_timeline(
    times: np.ndarray,
    masks: StateMasks,
    window_sec: float = WINDOW_SEC,
    output_path: str = "cognitive_states_timeline.png",
    figsize: tuple[float, float] = (14, 9),
) -> str:
    """
    Plot stacked binary state traces over task time.

    Each panel shows when the participant is in vs out of a cognitive state.
    Flow state is the topmost panel.  System 1/2 uses a System 2 vs System 1
    y-axis instead of in/out.
    """
    import matplotlib.pyplot as plt

    n_panels = len(TIMELINE_PANELS)
    fig, axes = plt.subplots(
        n_panels, 1, sharex=True, figsize=figsize, facecolor="white",
    )
    if n_panels == 1:
        axes = [axes]

    active_color   = "#2E86AB"
    inactive_color = "#D3D3D3"
    ambiguous_color = "#B0B0B0"
    system_values  = _system_timeline_values(masks)

    for ax, (key, title, y_on, y_off) in zip(axes, TIMELINE_PANELS):
        half = window_sec / 2.0

        if key == "system":
            for t, val in zip(times, system_values):
                if np.isnan(val):
                    ax.plot(
                        [t - half, t + half], [0.5, 0.5],
                        color=ambiguous_color, linewidth=3, solid_capstyle="butt",
                    )
                else:
                    color = active_color if val else "#E07A5F"
                    ax.plot(
                        [t - half, t + half], [val, val],
                        color=color, linewidth=3, solid_capstyle="butt",
                    )
        else:
            active = masks[key].astype(float)
            for t, on in zip(times, active):
                color = active_color if on else inactive_color
                ax.plot(
                    [t - half, t + half], [on, on],
                    color=color, linewidth=3, solid_capstyle="butt",
                )

        ax.set_ylim(-0.15, 1.15)
        ax.set_yticks([0, 1])
        ax.set_yticklabels([y_off, y_on], fontsize=9)
        ax.set_ylabel(title, fontsize=10, fontweight="bold")
        ax.grid(axis="x", alpha=0.25, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel("Time (s)", fontsize=11)
    fig.suptitle(
        "Cognitive States Over Time",
        fontsize=14, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ── dDTF snapshots at detected moments ────────────────────────────────────────

def _crop_segment(raw: mne.io.RawArray, centre_sec: float,
                  window_sec: float) -> mne.io.RawArray:
    """Crop a centred segment, clamped to recording bounds."""
    half  = window_sec / 2.0
    tmin  = max(0.0, centre_sec - half)
    tmax  = min(raw.times[-1], centre_sec + half)
    return raw.copy().crop(tmin=tmin, tmax=tmax)


def generate_state_ddtf_diagrams(
    task_raw: mne.io.RawArray,
    onsets: dict[str, StateOnsetResult],
    output_dir: str = "state_snapshots",
    ddtf_window_sec: float = DDTF_WINDOW_SEC,
    threshold: float = 0.25,
) -> dict[str, str | None]:
    """
    For each cognitive state with a detected onset, compute dDTF on a centred
    segment and save all-band + combined-band network diagrams.
    """
    os.makedirs(output_dir, exist_ok=True)
    paths: dict[str, str | None] = {}

    for state_key, onset in onsets.items():
        t = onset["first_onset_sec"]
        if t is None:
            paths[state_key] = None
            continue

        segment = _crop_segment(task_raw, t, ddtf_window_sec)
        ddtf    = compute_ddtf_connectivity(segment)

        label = state_key.replace("_", " ").title()
        t_str = f"{t:.0f}s"

        all_path = os.path.join(output_dir, f"ddtf_{state_key}_{t_str}_all.png")
        bands_path = os.path.join(output_dir, f"ddtf_{state_key}_{t_str}_bands.png")

        fig = plot_ddtf(
            ddtf, band="all", threshold=threshold,
            title=f"{label} @ {t:.1f} s  (all bands)",
        )
        fig.savefig(all_path, dpi=150, bbox_inches="tight")
        plt_close(fig)

        fig = plot_all_bands(
            ddtf, threshold=threshold,
            suptitle=f"{label} @ {t:.1f} s",
        )
        fig.savefig(bands_path, dpi=150, bbox_inches="tight")
        plt_close(fig)

        paths[state_key] = bands_path

    return paths


def plt_close(fig) -> None:
    """Close figure without requiring matplotlib at import time."""
    import matplotlib.pyplot as plt
    plt.close(fig)


# ── Full pipeline + report ────────────────────────────────────────────────────

def run_temporal_analysis(
    task_raw: mne.io.RawArray,
    baseline_flow: dict,
    baseline_sys12: dict,
    baseline_attn: dict,
    baseline_val: dict,
    task_val: dict,
    output_dir: str = "state_snapshots",
    window_sec: float = WINDOW_SEC,
    ddtf_window_sec: float = DDTF_WINDOW_SEC,
    threshold: float = 0.25,
) -> TemporalAnalysisResult:
    """
    Detect cognitive-state onsets during the task and generate dDTF diagrams
    at each first-onset time.
    """
    times, onsets, masks = detect_state_onsets(
        task_raw,
        baseline_flow, baseline_sys12, baseline_attn, baseline_val, task_val,
        window_sec=window_sec,
    )

    ddtf_paths = generate_state_ddtf_diagrams(
        task_raw, onsets,
        output_dir=output_dir,
        ddtf_window_sec=ddtf_window_sec,
        threshold=threshold,
    )

    timeline_path = plot_state_timeline(
        times, masks,
        window_sec=window_sec,
        output_path=os.path.join(output_dir, "cognitive_states_timeline.png"),
    )

    return TemporalAnalysisResult(
        window_sec=window_sec,
        ddtf_window_sec=ddtf_window_sec,
        times=times,
        masks=masks,
        onsets=onsets,
        ddtf_paths=ddtf_paths,
        timeline_path=timeline_path,
    )


_REPORT_RULE = "-" * 72


def _write_report(text: str, path: str) -> str:
    """Write report text to disk (UTF-8) and return the path written."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def format_temporal_report(result: TemporalAnalysisResult) -> str:
    """Build the temporal analysis report as plain text."""
    sep = "=" * 72
    lines = [
        "",
        sep,
        "  TEMPORAL COGNITIVE STATE ANALYSIS",
        sep,
        f"  Detection window : {result['window_sec']:.1f} s",
        f"  dDTF snapshot    : {result['ddtf_window_sec']:.1f} s (centred on onset)",
        "",
        _REPORT_RULE,
        "  STATE TIMING",
        _REPORT_RULE,
    ]

    for _key, onset in result["onsets"].items():
        if onset["first_onset_sec"] is None:
            lines.append(f"  {onset['state']}: never reached during task")
            continue

        lines.append(f"  {onset['state']}:")
        lines.append(f"      First reached       : {onset['first_onset_sec']:.1f} s")
        streak = onset["longest_streak"]
        if streak:
            lines.append(
                f"      Longest consecutive : {streak['duration_sec']:.1f} s  "
                f"({streak['start_sec']:.1f} s -> {streak['end_sec']:.1f} s, "
                f"{streak['n_windows']} windows)"
            )
        pct = 100.0 * onset["n_windows_active"] / onset["n_windows_total"]
        lines.append(
            f"      Total active        : {onset['n_windows_active']}/"
            f"{onset['n_windows_total']} windows ({pct:.0f}% of task)"
        )

    lines.extend([
        "",
        _REPORT_RULE,
        "  STATE TIMELINE PLOT",
        _REPORT_RULE,
    ])
    if result.get("timeline_path"):
        lines.append(f"  Saved -> {result['timeline_path']}")

    lines.extend([
        "",
        _REPORT_RULE,
        "  dDTF NETWORK DIAGRAMS",
        _REPORT_RULE,
    ])
    for key, path in result["ddtf_paths"].items():
        if path:
            lines.append(f"  {key:<18s} -> {path}")
        else:
            lines.append(f"  {key:<18s} -> (state not reached -- no diagram)")

    lines.extend(["", sep, ""])
    return "\n".join(lines)


def print_temporal_report(
    result: TemporalAnalysisResult,
    output_path: str | None = None,
) -> str | None:
    """Print temporal analysis report; optionally save to a text file."""
    text = format_temporal_report(result)
    print(text, end="")
    if output_path:
        return _write_report(text, output_path)
    return None


# Display order for cross-session temporal comparison.
TEMPORAL_COMPARE_STATE_ORDER = [
    "flow_state",
    "cognitive_load",
    "attention",
    "system_2",
    "system_1",
    "valence",
]


def _temporal_winner(
    val_a: float,
    val_b: float,
    label_a: str,
    label_b: str,
) -> str:
    """Return which session wins a numeric comparison, or 'Tie'."""
    if val_a > val_b:
        return label_a
    if val_b > val_a:
        return label_b
    return "Tie"


def format_temporal_comparison_report(
    result_a: TemporalAnalysisResult,
    result_b: TemporalAnalysisResult,
    label_a: str = "Task 1",
    label_b: str = "Task 2",
) -> str:
    """Build the cross-task temporal comparison report as plain text."""
    sep = "=" * 72
    lines = [
        "",
        sep,
        "  TEMPORAL ANALYSIS COMPARISON",
        sep,
        f"  {label_a} vs {label_b}",
        "",
        _REPORT_RULE,
        f"  {'State':<28s}  {'More active windows':<22s}  {'Longer consecutive'}",
        _REPORT_RULE,
    ]

    for key in TEMPORAL_COMPARE_STATE_ORDER:
        onset_a = result_a["onsets"][key]
        onset_b = result_b["onsets"][key]
        state_name = onset_a["state"]

        active_a = onset_a["n_windows_active"]
        active_b = onset_b["n_windows_active"]
        pct_a = 100.0 * active_a / onset_a["n_windows_total"] if onset_a["n_windows_total"] else 0.0
        pct_b = 100.0 * active_b / onset_b["n_windows_total"] if onset_b["n_windows_total"] else 0.0
        active_winner = _temporal_winner(active_a, active_b, label_a, label_b)

        streak_a = onset_a["longest_streak"]
        streak_b = onset_b["longest_streak"]
        dur_a = streak_a["duration_sec"] if streak_a else 0.0
        dur_b = streak_b["duration_sec"] if streak_b else 0.0
        win_a = streak_a["n_windows"] if streak_a else 0
        win_b = streak_b["n_windows"] if streak_b else 0

        if dur_a > dur_b or (dur_a == dur_b and win_a > win_b):
            streak_winner = label_a
        elif dur_b > dur_a or (dur_b == dur_a and win_b > win_a):
            streak_winner = label_b
        else:
            streak_winner = "Tie"

        active_detail = (
            f"{active_winner} ({active_a} vs {active_b} windows, "
            f"{pct_a:.0f}% vs {pct_b:.0f}%)"
        )
        streak_detail = (
            f"{streak_winner} ({dur_a:.1f} s / {win_a} win vs "
            f"{dur_b:.1f} s / {win_b} win)"
        )

        lines.append(f"  {state_name:<28s}  {active_detail:<22s}  {streak_detail}")

    lines.extend(["", sep, ""])
    return "\n".join(lines)


def print_temporal_comparison_report(
    result_a: TemporalAnalysisResult,
    result_b: TemporalAnalysisResult,
    label_a: str = "Task 1",
    label_b: str = "Task 2",
    output_path: str | None = None,
) -> str | None:
    """
    Compare temporal analyses from two tasks side by side.

    For each cognitive state, reports which task had more active windows and
    which had the longer longest consecutive streak.
    """
    text = format_temporal_comparison_report(result_a, result_b, label_a, label_b)
    print(text, end="")
    if output_path:
        return _write_report(text, output_path)
    return None
