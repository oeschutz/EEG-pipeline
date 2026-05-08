import mne

LEFT_FRONTAL_CHS = ["F1"]
ALPHA_BAND       = (8.0, 13.0)


def compute_left_frontal_alpha(raw: mne.io.RawArray) -> float:
    """
    Compute left frontal alpha power (8–13 Hz) at F1.

    Left frontal alpha is commonly used as a marker of approach motivation
    and emotional valence via frontal alpha asymmetry (FAA = F4 - F3 alpha,
    or here F2 - F1). Higher left frontal alpha indicates relative left
    hypoactivation.

    Parameters
    ----------
    raw : mne.io.RawArray
        Cleaned, bandpass-filtered EEG.

    Returns
    -------
    float
        Alpha power at F1 in µV²/Hz.
    """
    raw_f1 = raw.copy().pick_channels(LEFT_FRONTAL_CHS)
    sfreq  = raw_f1.info["sfreq"]
    n_fft  = int(4 * sfreq)

    spectrum    = raw_f1.compute_psd(method="welch", n_fft=n_fft, fmin=1.0, fmax=40.0)
    psds, freqs = spectrum.get_data(return_freqs=True)   # (1, n_freqs)

    mask = (freqs >= ALPHA_BAND[0]) & (freqs < ALPHA_BAND[1])
    return float(psds[0, mask].mean())