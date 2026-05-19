import re
import numpy as np
import pandas as pd
import mne
from pathlib import Path
import matplotlib.pyplot as plt
from mne.preprocessing import ICA

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD OPENBCI TEXT FILE
# ─────────────────────────────────────────────────────────────────────────────

def load_galea_txt(filepath: str, CH_NAMES: list, SFREQ: int) -> mne.io.RawArray:
    """
    Parses the standard OpenBCI GUI raw EXG .txt format:
      %OpenBCI Raw EXG Data
      %Number of channels = 8
      ...header comments...
      Sample Index, EXG Channel 0, EXG Channel 1, ..., Timestamp
    Returns an MNE RawArray (µV, scaled for MNE's internal V storage).
    """
    filepath = Path(filepath)
    header_lines = 0

    with open(filepath, "r") as f:
        for line in f:
            if line.startswith("%"):
                header_lines += 1
            else:
                break

    df = pd.read_csv(filepath, skiprows=header_lines)
    df.columns = df.columns.str.strip()

    # ── Trim to marker-8 region ───────────────────────────────────────────────
    marker_col = next((c for c in df.columns if "marker" in c.lower()), None)
    if marker_col is not None:
        print("Trimming to region between markers")
        marker_hits = df.index[df[marker_col].astype(str).str.strip() == "8.0"].tolist()
        #print(marker_hits)
        if len(marker_hits) >= 2:
            df = df.loc[marker_hits[0] : marker_hits[-1]].reset_index(drop=True)
            #print("length of df: ", len(df))
        elif len(marker_hits) == 1:
            df = df.loc[marker_hits[0] :].reset_index(drop=True)
            #print("length of df: ", len(df))
        # If no marker-8 rows exist, fall through and use the whole file


    # Keep only EXG columns
    eeg_cols = [c for c in df.columns if re.search(r"EEG Channel|EOG Channel", c, re.I)]
    if not eeg_cols:
        # Fallback: columns 1..N-1 (skip sample index, drop timestamp)
        eeg_cols = df.columns[1:-1].tolist()

    data = df[eeg_cols].values.T.astype(np.float64)

    # Trim CH_NAMES to however many channels are actually in the file
    n_ch = data.shape[0]
    if n_ch < len(CH_NAMES):
        ch_names = CH_NAMES[:n_ch]
    elif n_ch > len(CH_NAMES):
        ch_names = CH_NAMES
        print(ch_names)
        print(len(data))
        data = np.delete(data, np.s_[len(CH_NAMES):n_ch], 0)
        n_ch = data.shape[0]
        print(n_ch)
    else:
        ch_names = CH_NAMES

    # OpenBCI GUI outputs µV already in .txt — convert to V for MNE
    data_V = data * 1e-6

    info = mne.create_info(ch_names=ch_names, sfreq=SFREQ, ch_types="eeg")
    raw = mne.io.RawArray(data_V, info, verbose=False)
    raw.set_montage("standard_1020", match_case=False, on_missing="warn")
    return raw

def load_baseline_galea(filepath: str, CH_NAMES: list, SFREQ: int) -> list:
    """
    loads the baseline file into 3 separate raw objects (one for each baseline task)
    """

    filepath = Path(filepath)
    header_lines = 0

    with open(filepath, "r") as f:
        for line in f:
            if line.startswith("%"):
                header_lines += 1
            else:
                break

    df = pd.read_csv(filepath, skiprows=header_lines)
    df.columns = df.columns.str.strip()

    # ── Trim to marker-8 region ───────────────────────────────────────────────
    marker_col = next((c for c in df.columns if "marker" in c.lower()), None)
    if marker_col is not None:
        print("Trimming to region between markers")
        marker_hits = df.index[df[marker_col].astype(str).str.strip() == "8.0"].tolist()
        #print(marker_hits)
        if len(marker_hits) >= 2:
            df = df.loc[marker_hits[0] : marker_hits[-1]].reset_index(drop=True)
            #print("length of df: ", len(df))
        elif len(marker_hits) == 1:
            df = df.loc[marker_hits[0] :].reset_index(drop=True)
            #print("length of df: ", len(df))
        # If no marker-8 rows exist, fall through and use the whole file

    # ── Divide into baselining tasks ──────────────────────────────────────────
    baseline_data = [df.loc[2500:27500], df.loc[32500:57500], df.loc[62500:87500], df[27500:32500], df[62500:87500], df]
    baseline_raw = []
    for df in baseline_data:
        # Keep only EXG columns
        eeg_cols = [c for c in df.columns if re.search(r"EEG Channel|EOG Channel", c, re.I)]
        if not eeg_cols:
            # Fallback: columns 1..N-1 (skip sample index, drop timestamp)
            eeg_cols = df.columns[1:-1].tolist()

        data = df[eeg_cols].values.T.astype(np.float64)

        # Trim CH_NAMES to however many channels are actually in the file
        n_ch = data.shape[0]
        if n_ch < len(CH_NAMES):
            ch_names = CH_NAMES[:n_ch]
        elif n_ch > len(CH_NAMES):
            ch_names = CH_NAMES
            print(ch_names)
            print(len(data))
            data = np.delete(data, np.s_[len(CH_NAMES):n_ch], 0)
            n_ch = data.shape[0]
            print(n_ch)
        else:
            ch_names = CH_NAMES

        # OpenBCI GUI outputs µV already in .txt — convert to V for MNE
        data_V = data * 1e-6

        info = mne.create_info(ch_names=ch_names, sfreq=SFREQ, ch_types="eeg")
        raw = mne.io.RawArray(data_V, info, verbose=False)
        raw.set_montage("standard_1020", match_case=False, on_missing="warn")
        baseline_raw.append(raw)
    return baseline_raw


def filter_data(raw: mne.io.RawArray) -> mne.io.RawArray:
    """Bandpass filter + notch (50 Hz & 60 Hz power line frequencies)."""
    raw = raw.copy()
    raw.filter(l_freq=0.5, h_freq=40.0, fir_design="firwin", verbose=False)
    raw.notch_filter(freqs=[50.0, 60.0], verbose=False)
    return raw



# remove_blinks ideally needs the EOG channels to be good - otherwise, can maybe bring eye tracking data in, but would need to manually sync the eye tracking to galea data
def remove_blinks(raw: mne.io.RawArray) -> mne.io.RawArray: 
    """
    Remove eye-blink artifacts from an EEG signal using ICA.

    The function fits an ICA decomposition on the raw signal, automatically
    identifies the component(s) most correlated with blink activity (via EOG
    channel detection or a frontal-electrode heuristic when no EOG channel is
    present), excludes those components, and returns a cleaned copy of the data.

    Parameters
    ----------
    raw : mne.io.RawArray
        Continuous raw EEG data. Should already be filtered (e.g. 1–40 Hz) for
        best ICA performance.

    Returns
    -------
    mne.io.RawArray
        A new RawArray with blink components removed. The original object is
        not modified.
    """
    raw = raw.copy()

    # ------------------------------------------------------------------ #
    # 1.  Fit ICA                                                          #
    # ------------------------------------------------------------------ #
    n_channels = len(raw.pick_types(eeg=True, exclude="bads").ch_names)
    n_components = min(20, n_channels)          # 20 components is usually enough
                                                # to capture blink variance while
                                                # keeping the decomposition stable.

    ica = ICA(
        n_components=n_components,
        method="fastica",                       # FastICA is fast and well-tested
        max_iter="auto",
        random_state=42,                        # reproducibility
    )
    ica.fit(raw, picks="eeg", reject_by_annotation=True)

    # ------------------------------------------------------------------ #
    # 2.  Identify blink component(s)                                      #
    # ------------------------------------------------------------------ #
    eog_channels = mne.pick_types(raw.info, eog=True)

    if len(eog_channels) > 0:
        # Preferred path: correlate ICA sources with the EOG channel(s).
        eog_indices, eog_scores = ica.find_bads_eog(raw)
        blink_indices = eog_indices
    else:
        # Fallback: use a frontal EEG electrode as a blink proxy.
        # Blinks produce large, slow deflections on frontal electrodes, so the
        # ICA component that correlates most with Fp1 or Fp2 is a reasonable
        # surrogate for EOG.
        frontal_candidates = ["Fp1", "Fp2", "AF3", "AF4", "FP1", "FP2"]
        proxy_ch = next(
            (ch for ch in frontal_candidates if ch in raw.ch_names), None
        )

        if proxy_ch is None:
            # Last resort: pick the channel with the highest peak-to-peak
            # amplitude on the assumption that blink contamination dominates.
            data, _ = raw.get_data(picks="eeg", return_times=True)
            ptp = np.ptp(data, axis=1)
            proxy_ch = raw.ch_names[int(np.argmax(ptp))]

        blink_indices, _ = ica.find_bads_eog(raw, ch_name=proxy_ch)

        if not blink_indices:
            # If the correlation-based search found nothing, fall back to
            # selecting the component with the largest absolute frontal weight.
            proxy_idx = raw.ch_names.index(proxy_ch)
            frontal_weights = np.abs(ica.get_components()[proxy_idx, :])
            blink_indices = [int(np.argmax(frontal_weights))]

    # ------------------------------------------------------------------ #
    # 3.  Remove blink components and reconstruct the signal               #
    # ------------------------------------------------------------------ #
    ica.exclude = blink_indices
    raw_clean = raw.copy()
    ica.apply(raw_clean)               # subtracts excluded components in-place

    return raw_clean

import mne
import numpy as np

# use this to show segment of cognitive states change throughout the task
def segment_raw(raw: mne.io.RawArray, epoch_duration: float = 60.0) -> list[mne.io.RawArray]:
    """
    Segment a continuous RawArray into fixed-length 60 second RawArray objects.

    Any trailing samples that do not fill a complete epoch are discarded.

    Parameters
    ----------
    raw             : mne.io.RawArray
        Continuous raw EEG/EOG data.
    epoch_duration  : float
        Length of each segment in seconds (default: 2.0).

    Returns
    -------
    list[mne.io.RawArray]
        Ordered list of non-overlapping raw segments, each exactly
        `epoch_duration` seconds long and sharing the same Info object.
    """
    sfreq      = raw.info["sfreq"]
    epoch_samp = int(epoch_duration * sfreq)   # samples per epoch
    data       = raw.get_data()                # (n_channels, n_times)
    n_times    = data.shape[1]
    n_epochs   = n_times // epoch_samp         # discard incomplete tail

    segments = [
        mne.io.RawArray(
            data[:, i * epoch_samp : (i + 1) * epoch_samp],
            raw.info,
            verbose=False,
        )
        for i in range(n_epochs)
    ]

    return segments

