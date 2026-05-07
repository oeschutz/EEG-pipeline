import mne
import numpy as np
from typing import TypedDict

PARIETAL_CHS = ["P3", "P4", "Pz"]
FRONTAL_CHS  = ["F1", "F2"]
THETA_BAND   = (4.0,  8.0)
ALPHA_BAND   = (8.0, 13.0)


def compute_sys12_features(raw: mne.io.RawArray) -> dict[str, float]:
    """
    Compute parietal alpha (8–13 Hz) and frontal theta (4–8 Hz) power.

    Frontal theta (F1, F2) increases linearly with working memory load.
    Parietal alpha (P3, P4, Pz) is suppressed during attentional engagement
    and is a reliable inverse marker of cognitive effort.

    Parameters
    ----------
    raw : mne.io.RawArray
        Cleaned, bandpass-filtered EEG.

    Returns
    -------
    dict with keys:
        parietal_alpha  – mean alpha power over P3, P4, Pz  (µV²/Hz)
        frontal_theta   – mean theta power over F1, F2      (µV²/Hz)
    """
    target_chs = list(dict.fromkeys(PARIETAL_CHS + FRONTAL_CHS))  # preserve order, no dupes
    raw_subset = raw.copy().pick_channels(target_chs)

    sfreq  = raw_subset.info["sfreq"]
    n_fft  = int(4 * sfreq)

    spectrum   = raw_subset.compute_psd(method="welch", n_fft=n_fft, fmin=1.0, fmax=40.0)
    psds, freqs = spectrum.get_data(return_freqs=True)   # (n_ch, n_freqs)
    ch_names   = raw_subset.ch_names

    def _mean_band_power(channels: list[str], fmin: float, fmax: float) -> float:
        idx  = [ch_names.index(ch) for ch in channels if ch in ch_names]
        mask = (freqs >= fmin) & (freqs < fmax)
        return float(psds[np.ix_(idx, mask)].mean())

    return {
        "parietal_alpha": _mean_band_power(PARIETAL_CHS, *ALPHA_BAND),
        "frontal_theta":  _mean_band_power(FRONTAL_CHS,  *THETA_BAND),
    }

