import mne
import numpy as np
from typing import TypedDict


# ── Channel layout ─────────────────────────────────────────────────────────────
# Grouped by scalp region for the spatial weighting step.
# The QStates paper weights frontal and parietal regions most heavily for
# cognitive load; central electrodes contribute at a lower weight.
FRONTAL_CHS   = ["F1", "F2"]
CENTRAL_CHS   = ["C3", "C4", "Cz"]
PARIETAL_CHS  = ["P3", "P4", "Pz"]
OCCIPITAL_CHS = ["O1", "O2"]
EEG_CHANNELS  = ["F1", "F2", "C3", "C4", "P3", "P4", "O1", "O2", "Cz", "Pz"]

# Spatial weights per region, matching the QStates emphasis on
# frontal and parieto-occipital regions for the beta/alpha ratio.
# Frontal beta increases and parietal alpha suppression are the two
# strongest correlates of cognitive load in this framework.
REGION_WEIGHTS: dict[str, float] = {
    "frontal":   0.35,
    "central":   0.15,
    "parietal":  0.35,
    "occipital": 0.15,
}

# Frequency bands
ALPHA_BAND = (8.0,  13.0)
BETA_BAND  = (13.0, 30.0)

# QStates output window: 2-second epochs, updated every 2 seconds (no overlap)
WINDOW_SEC  = 2.0
OVERLAP_SEC = 0.0


class BetaAlphaResult(TypedDict):
    ratio_per_window:   np.ndarray   # (n_windows,)  scalar ratio per epoch
    ratio_per_channel:  np.ndarray   # (n_windows, n_eeg_ch)  per-channel ratios
    times:              np.ndarray   # (n_windows,)  centre time of each window [s]
    mean_ratio:         float        # grand mean across all windows
    ch_names:           list[str]


def _band_power(psd: np.ndarray, freqs: np.ndarray,
                fmin: float, fmax: float) -> float:
    """
    Mean power spectral density in [fmin, fmax) for a single channel.

    Parameters
    ----------
    psd   : (n_freqs,) PSD in µV²/Hz
    freqs : (n_freqs,) frequency axis

    Returns
    -------
    Scalar mean power (µV²/Hz).
    """
    mask = (freqs >= fmin) & (freqs < fmax)
    if not mask.any():
        raise ValueError(f"No PSD bins in [{fmin}, {fmax}) Hz.")
    return float(psd[mask].mean())


def _weighted_ratio(ratios: np.ndarray, ch_names: list[str]) -> float:
    """
    Combine per-channel beta/alpha ratios into a single scalar using the
    region-based spatial weighting scheme from QStates.

    Channels are grouped by scalp region; each group's mean ratio is
    weighted and summed. This emphasises frontal beta increases and
    parietal alpha suppression — the two dominant signatures of cognitive
    load in the QStates framework.

    Parameters
    ----------
    ratios   : (n_channels,) per-channel beta/alpha ratios
    ch_names : channel labels corresponding to `ratios`

    Returns
    -------
    Single spatially-weighted beta/alpha scalar.
    """
    region_map = {
        "frontal":   FRONTAL_CHS,
        "central":   CENTRAL_CHS,
        "parietal":  PARIETAL_CHS,
        "occipital": OCCIPITAL_CHS,
    }

    weighted_sum  = 0.0
    weight_total  = 0.0

    for region, members in region_map.items():
        idx = [ch_names.index(ch) for ch in members if ch in ch_names]
        if not idx:
            continue
        region_mean    = ratios[idx].mean()
        w              = REGION_WEIGHTS[region]
        weighted_sum  += w * region_mean
        weight_total  += w

    # Normalise in case some regions are missing channels
    return weighted_sum / weight_total if weight_total > 0 else float(ratios.mean())


def compute_beta_alpha_ratio(
    raw: mne.io.RawArray,
    window_sec:  float = WINDOW_SEC,
    overlap_sec: float = OVERLAP_SEC,
) -> BetaAlphaResult:
    """
    Compute the beta/alpha power ratio as described in McDonald & Soussou (2011).

    The ratio β/α is a well-validated index of cognitive engagement and
    workload: beta power (13–30 Hz) increases with active cognitive processing
    while alpha power (8–13 Hz) is suppressed, so their ratio rises with
    cognitive load.

    The computation follows the QStates methodology:
      1. Segment the continuous signal into non-overlapping 2-second windows.
      2. Estimate the PSD of each window with Welch's method.
      3. Compute β/α per channel.
      4. Combine channels into a single scalar via regional spatial weighting,
         emphasising frontal and parietal electrodes.

    Parameters
    ----------
    raw         : mne.io.RawArray
        Cleaned, bandpass-filtered (0.5–40 Hz) EEG. EOG channels are ignored.
    window_sec  : float
        Epoch length in seconds (default: 2.0, matching QStates output rate).
    overlap_sec : float
        Window overlap in seconds (default: 0.0 — non-overlapping, as in
        QStates). Use e.g. 1.0 for a 50% overlap if higher temporal resolution
        is needed.

    Returns
    -------
    BetaAlphaResult
        ratio_per_window  : (n_windows,) spatially-weighted β/α per epoch.
                            Higher values → greater cognitive engagement.
        ratio_per_channel : (n_windows, n_eeg_ch) β/α at each electrode,
                            useful for topographic analysis.
        times             : (n_windows,) centre time of each epoch in seconds.
        mean_ratio        : grand mean β/α across the recording.
        ch_names          : EEG channel labels (same order as ratio_per_channel).
    """
    # ── Extract EEG-only data ─────────────────────────────────────────────────
    raw_eeg  = raw.copy().pick_channels(EEG_CHANNELS)
    data     = raw_eeg.get_data()              # (n_ch, n_times) µV
    sfreq    = raw_eeg.info["sfreq"]
    ch_names = raw_eeg.ch_names
    n_ch     = len(ch_names)

    win_samp  = int(window_sec  * sfreq)
    step_samp = int((window_sec - overlap_sec) * sfreq)
    n_times   = data.shape[1]

    # ── Build epoch start indices ─────────────────────────────────────────────
    starts = np.arange(0, n_times - win_samp + 1, step_samp)
    n_win  = len(starts)

    ratio_per_channel = np.zeros((n_win, n_ch))
    times             = np.zeros(n_win)

    # Welch n_fft: use the full window length for maximum frequency resolution.
    # For 2-second windows, Δf = sfreq / win_samp ≈ 0.5 Hz — sufficient to
    # resolve both bands with a clean boundary at 13 Hz.
    n_fft = win_samp

    for wi, start in enumerate(starts):
        epoch = data[:, start : start + win_samp]   # (n_ch, win_samp)
        times[wi] = (start + win_samp / 2) / sfreq  # centre time

        for ci in range(n_ch):
            # Per-channel Welch PSD
            spectrum = mne.time_frequency.psd_array_welch(
                epoch[ci : ci + 1],          # (1, win_samp)
                sfreq   = sfreq,
                fmin    = ALPHA_BAND[0],
                fmax    = BETA_BAND[1],
                n_fft   = n_fft,
                verbose = False,
            )
            psd, freqs = spectrum[0][0], spectrum[1]  # (n_freqs,), (n_freqs,)

            alpha_pw = _band_power(psd, freqs, *ALPHA_BAND)
            beta_pw  = _band_power(psd, freqs, *BETA_BAND)

            # Guard against a silent alpha floor to avoid divide-by-zero.
            # A very small alpha (< 0.01 µV²/Hz) can occur in heavily
            # artifact-contaminated epochs; the ratio is capped at 100
            # rather than left as ±inf.
            ratio_per_channel[wi, ci] = (
                beta_pw / alpha_pw if alpha_pw > 0.01 else 100.0
            )

    # ── Spatially-weighted scalar per window ──────────────────────────────────
    ratio_per_window = np.array([
        _weighted_ratio(ratio_per_channel[wi], ch_names)
        for wi in range(n_win)
    ])

    return BetaAlphaResult(
        ratio_per_window  = ratio_per_window,
        ratio_per_channel = ratio_per_channel,
        times             = times,
        mean_ratio        = float(ratio_per_window.mean()),
        ch_names          = ch_names,
    )

