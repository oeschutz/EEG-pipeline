import mne
import numpy as np
from typing import TypedDict


# Channel name constants
EOG_CHANNELS  = ["EOG 1", "EOG 2"]
EEG_CHANNELS  = ["F1", "F2", "C3", "C4", "P3", "P4", "O1", "O2", "Cz", "Pz"]
FRONTAL_CHS   = ["F1", "F2"]
RT_CENTRAL_CH = ["C4"]                  # right central (10-20 convention)

# Frequency band definitions (Hz)
THETA_BAND = (4.0,  8.0)
ALPHA_BAND = (8.0, 13.0)
BETA_BAND  = (13.0, 30.0)


class BandPowerFeatures(TypedDict):
    frontal_theta:        float   # mean theta power over F1, F2
    frontal_alpha:        float   # mean alpha power over F1, F2
    right_central_alpha:  float   # alpha power at C4
    theta_power:          float   # mean theta power across all EEG channels
    alpha_power:          float   # mean alpha power across all EEG channels
    beta_power:           float   # mean beta power  across all EEG channels


def _band_power(psds: np.ndarray, freqs: np.ndarray,
                fmin: float, fmax: float) -> np.ndarray:
    """
    Compute mean power in [fmin, fmax) for each channel.

    Uses rectangular (mean) integration over the PSD bins that fall inside
    the requested band — equivalent to trapezoidal integration when the
    frequency resolution is fine and uniform, but cheaper.

    Parameters
    ----------
    psds  : (n_channels, n_freqs) array of power values (µV²/Hz).
    freqs : (n_freqs,) array of centre frequencies for each PSD bin.

    Returns
    -------
    (n_channels,) array of mean band power values.
    """
    band_mask = (freqs >= fmin) & (freqs < fmax)
    if not band_mask.any():
        raise ValueError(
            f"No frequency bins found in [{fmin}, {fmax}) Hz. "
            f"Frequency resolution is {freqs[1] - freqs[0]:.3f} Hz; "
            "consider using a longer epoch."
        )
    return psds[:, band_mask].mean(axis=1)


def compute_flow_features(raw: mne.io.RawArray) -> BandPowerFeatures:
    """
    Compute frequency-band power features from a 12-channel EEG/EOG recording.

    Expected channel layout
    -----------------------
    EOG : "EOG 1", "EOG 2"
    EEG : "F1", "F2", "C3", "C4", "P3", "P4", "O1", "O2", "Cz", "Pz"

    Features returned
    -----------------
    frontal_theta       – mean θ power (4–8 Hz)   at F1, F2
    frontal_alpha       – mean α power (8–13 Hz)  at F1, F2
    right_central_alpha – α power (8–13 Hz)       at C4
    theta_power         – mean θ power across all 10 EEG channels
    alpha_power         – mean α power across all 10 EEG channels
    beta_power          – mean β power (13–30 Hz) across all 10 EEG channels

    Parameters
    ----------
    raw : mne.io.RawArray
        Continuous or epoched raw data.  Should already be band-pass filtered
        (e.g. 0.5–45 Hz) and have blink/muscle artifacts removed.

    Returns
    -------
    BandPowerFeatures
        Dictionary of scalar power estimates (µV²/Hz, averaged within band).
    """
    # ------------------------------------------------------------------ #
    # 1.  Compute PSD over EEG channels only (drop EOG)                   #
    # ------------------------------------------------------------------ #
    raw_eeg = raw.copy().pick_channels(EEG_CHANNELS)

    # Welch PSD — n_fft controls frequency resolution (Δf = sfreq / n_fft).
    # 4-second windows (n_fft = 4 * sfreq) give Δf ≈ 0.25 Hz, which is
    # fine enough to resolve all three bands without excessive variance.
    sfreq  = raw_eeg.info["sfreq"]
    n_fft  = int(4 * sfreq)

    spectrum = raw_eeg.compute_psd(method="welch", n_fft=n_fft,
                                   fmin=0.5, fmax=40.0)
    psds, freqs = spectrum.get_data(return_freqs=True)   # (n_ch, n_freqs), µV²/Hz

    ch_names = raw_eeg.ch_names                          # preserves pick order

    # ------------------------------------------------------------------ #
    # 2.  Helper: extract rows for a named subset of channels             #
    # ------------------------------------------------------------------ #
    def _ch_idx(names: list[str]) -> list[int]:
        return [ch_names.index(ch) for ch in names]

    # ------------------------------------------------------------------ #
    # 3.  Per-band power vectors                                          #
    # ------------------------------------------------------------------ #
    theta_pw = _band_power(psds, freqs, *THETA_BAND)   # shape (10,)
    alpha_pw = _band_power(psds, freqs, *ALPHA_BAND)   # shape (10,)
    beta_pw  = _band_power(psds, freqs, *BETA_BAND)    # shape (10,)

    # ------------------------------------------------------------------ #
    # 4.  Aggregate into named features                                   #
    # ------------------------------------------------------------------ #
    f_idx  = _ch_idx(FRONTAL_CHS)    # [F1, F2]
    rc_idx = _ch_idx(RT_CENTRAL_CH)  # [C4]

    return BandPowerFeatures(
        frontal_theta        = float(theta_pw[f_idx].mean()),
        frontal_alpha        = float(alpha_pw[f_idx].mean()),
        right_central_alpha  = float(alpha_pw[rc_idx].mean()),
        theta_power          = float(theta_pw.mean()),
        alpha_power          = float(alpha_pw.mean()),
        beta_power           = float(beta_pw.mean()),
    )

