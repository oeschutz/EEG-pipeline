import re
import numpy as np
import pandas as pd
import mne
from pathlib import Path
import matplotlib.pyplot as plt
from mne.preprocessing import ICA

EMG_CHANNELS = [1, 2, 3, 4, 7, 8]
EOG_CHANNELS = [5, 6]
EEG_CHANNELS = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18]

GALEA_SAMPLING_RATE = 250.0
CHANNEL_TYPE_LABELS = ['emg', 'emg', 'emg', 'emg',
                       'eog', 'eog',
                       'emg', 'emg',
                       'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
                       'eeg', 'eeg', 'eeg', 'eeg', 'eeg']

LINE_FREQ = 60.0

FREQ_BANDS = {"Delta": [0.5, 4],
               "Theta": [4, 8],
               "Alpha": [8, 13],
               "Beta": [13, 30],
               "Gamma": [30, 50],
               "High Gamma": [50, 100]}


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD OPENBCI TEXT FILE
# ─────────────────────────────────────────────────────────────────────────────

# def load_galea_txt(filepath: str, CH_NAMES: list, SFREQ: int) -> mne.io.RawArray:
#     """
#     Parses the standard OpenBCI GUI raw EXG .txt format:
#       %OpenBCI Raw EXG Data
#       %Number of channels = 8
#       ...header comments...
#       Sample Index, EXG Channel 0, EXG Channel 1, ..., Timestamp
#     Returns an MNE RawArray (µV, scaled for MNE's internal V storage).
#     """
#     filepath = Path(filepath)
#     header_lines = 0

#     with open(filepath, "r") as f:
#         for line in f:
#             if line.startswith("%"):
#                 header_lines += 1
#             else:
#                 break

#     df = pd.read_csv(filepath, skiprows=header_lines)
#     df.columns = df.columns.str.strip()

#     # ── Trim to marker-8 region ───────────────────────────────────────────────
#     marker_col = next((c for c in df.columns if "marker" in c.lower()), None)
#     if marker_col is not None:
#         print("Trimming to region between markers")
#         marker_hits = df.index[df[marker_col].astype(str).str.strip() == "8.0"].tolist()
#         #print(marker_hits)
#         if len(marker_hits) >= 2:
#             df = df.loc[marker_hits[0] : marker_hits[-1]].reset_index(drop=True)
#             #print("length of df: ", len(df))
#         elif len(marker_hits) == 1:
#             df = df.loc[marker_hits[0] :].reset_index(drop=True)
#             #print("length of df: ", len(df))
#         # If no marker-8 rows exist, fall through and use the whole file


#     # Keep only EEG and EOG columns
#     eeg_cols = [c for c in df.columns if re.search(r"EEG Channel|EOG Channel", c, re.I)]
#     if not eeg_cols:
#         # Fallback: columns 1..N-1 (skip sample index, drop timestamp)
#         eeg_cols = df.columns[1:-1].tolist()

#     data = df[eeg_cols].values.T.astype(np.float64)

#     # Trim CH_NAMES to however many channels are actually in the file
#     n_ch = data.shape[0]
#     if n_ch < len(CH_NAMES):
#         ch_names = CH_NAMES[:n_ch]
#     elif n_ch > len(CH_NAMES):
#         ch_names = CH_NAMES
#         print(ch_names)
#         print(len(data))
#         data = np.delete(data, np.s_[len(CH_NAMES):n_ch], 0)
#         n_ch = data.shape[0]
#         print(n_ch)
#     else:
#         ch_names = CH_NAMES

#     # OpenBCI GUI outputs µV already in .txt — convert to V for MNE
#     data_V = data * 1e-6

#     eeg_channel_locations = {s: s.split(" - ", 1)[1] for s in eeg_cols if "EEG" in s}
#     info = mne.create_info(ch_names=ch_names, sfreq=SFREQ, ch_types=["eog", "eog", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg"])
#     raw = mne.io.RawArray(data_V, info, verbose=False)
#     raw.rename_channels(eeg_channel_locations)
#     raw = raw.set_montage(make_galea_mne_montage(eeg_channel_locations))
#     return raw

TASK_TARGET_SEC     = 20 * 60   # 20 minutes
BASELINE_TARGET_SEC =  6 * 60   #  6 minutes


def _segment_duration_sec(segment: pd.DataFrame, sfreq: float) -> float:
    return len(segment) / sfreq


def _mask_between(exg_data: pd.DataFrame, start_index, end_index) -> pd.Series:
    return (exg_data.index >= start_index) & (exg_data.index <= end_index)


def _trim_galea_markers(
    exg_data: pd.DataFrame,
    marker_value: int = 8,
    recording_type: str = "task",
    sfreq: float = GALEA_SAMPLING_RATE,
) -> pd.DataFrame:
    """
    Trim a Galea EXG recording using marker events.

    Task recordings
    ---------------
    - 0 markers  : use the full file
    - 1 marker   : keep the longer of [file start → marker] or [marker → file end]
    - 2 markers  : keep rows between the first and second markers (inclusive)
    - 3+ markers : keep the marker pair whose span is closest to 20 minutes

    Baseline recordings
    -------------------
    - 0 markers  : use the full file
    - 1 marker   : keep whichever side of the marker is closest to 6 minutes
    - 2 markers  : keep rows between the first and second markers (inclusive)
    - 3+ markers : keep the marker pair whose span is closest to 6 minutes
    """
    marker_indices = exg_data.index[exg_data["Marker"] == marker_value].tolist()
    if not marker_indices:
        return exg_data.reset_index(drop=True)

    first_row = exg_data.index[0]
    last_row  = exg_data.index[-1]

    if len(marker_indices) == 1:
        marker = marker_indices[0]
        before = exg_data.loc[first_row:marker]
        after  = exg_data.loc[marker:last_row]

        if recording_type == "baseline":
            target = BASELINE_TARGET_SEC
            dur_before = _segment_duration_sec(before, sfreq)
            dur_after  = _segment_duration_sec(after, sfreq)
            mask = _mask_between(
                exg_data, first_row, marker
            ) if abs(dur_before - target) <= abs(dur_after - target) else (
                exg_data.index >= marker
            )
        else:
            dur_before = _segment_duration_sec(before, sfreq)
            dur_after  = _segment_duration_sec(after, sfreq)
            mask = _mask_between(
                exg_data, first_row, marker
            ) if dur_before >= dur_after else (exg_data.index >= marker)

    elif len(marker_indices) == 2:
        mask = _mask_between(exg_data, marker_indices[0], marker_indices[1])

    else:
        target = (
            BASELINE_TARGET_SEC if recording_type == "baseline"
            else TASK_TARGET_SEC
        )
        best_start, best_end = marker_indices[0], marker_indices[1]
        best_diff = float("inf")

        for i, start_idx in enumerate(marker_indices):
            for end_idx in marker_indices[i + 1:]:
                segment = exg_data.loc[start_idx:end_idx]
                diff = abs(_segment_duration_sec(segment, sfreq) - target)
                if diff < best_diff:
                    best_diff = diff
                    best_start, best_end = start_idx, end_idx

        mask = _mask_between(exg_data, best_start, best_end)

    return exg_data[mask].reset_index(drop=True)


def _galea_df_to_raw(exg_data: pd.DataFrame) -> mne.io.RawArray:
    """Build an MNE RawArray from a Galea EXG dataframe segment."""
    exg_cols = exg_data.columns[1:19].to_list()
    eeg_channel_locations = {
        s: s.split(" - ", 1)[1] for s in exg_cols if "EEG" in s
    }

    mne_info = mne.create_info(
        ch_names=exg_cols,
        sfreq=GALEA_SAMPLING_RATE,
        ch_types=CHANNEL_TYPE_LABELS,
    )

    galea_mne = mne.io.RawArray(exg_data[exg_cols].values.T, mne_info)
    galea_mne.rename_channels(eeg_channel_locations)
    galea_mne = galea_mne.set_montage(make_galea_mne_montage(eeg_channel_locations))
    return galea_mne


def load_galea_txt(filepath: str, CH_NAMES: list, SFREQ: int) -> mne.io.RawArray:
    exg_data = _trim_galea_markers(
        pd.read_csv(filepath, skiprows=4), recording_type="task"
    )
    galea_mne = _galea_df_to_raw(exg_data)
    fig = galea_mne.compute_psd(picks="eeg").plot(picks="eeg", show=False)
    plt.show()
    plt.clf()
    return galea_mne


def make_galea_mne_montage(eeg_channel_locations, verbose: bool = False) -> mne.channels.DigMontage:
    """
    Creates a montage for the Galea EEG device.

    Parameters
    ----------
    verbose : bool, optional
        When on, makes a plot of the montage on 10-20 plot.

    Returns
    -------
    mne.channels.DigMontage
        Montage for the Galea EEG device.
    """
    mont1020 = mne.channels.make_standard_montage('standard_1020')
    kept_channels = eeg_channel_locations.values()
    ind = [i for (i, channel) in enumerate(mont1020.ch_names) if channel in kept_channels]

    mont1020_galea = mont1020.copy()
    mont1020_galea.ch_names = [mont1020.ch_names[x] for x in ind]
    kept_channel_info = [mont1020.dig[x+3] for x in ind]
    mont1020_galea.dig = mont1020.dig[0:3]+kept_channel_info
    if verbose:
        mont1020_galea.plot()

    return mont1020_galea

def load_baseline_galea(filepath: str, CH_NAMES: list, SFREQ: int) -> list:
    """
    Load a baseline Galea recording into separate RawArrays per baseline task.

    Uses the same loading pipeline as load_galea_txt (skiprows=4, marker-8
    trimming, EXG columns 1–18, Galea montage), then splits the trimmed data
    into the existing baseline task segments.

    Returns
    -------
    list[mne.io.RawArray]
        [math, eyes, rest, sw1, sw2, full]
    """
    exg_data = _trim_galea_markers(
        pd.read_csv(filepath, skiprows=4), recording_type="baseline"
    )

    baseline_segments = [
        exg_data.loc[2500:27500],    # math
        exg_data.loc[32500:57500],   # eyes
        exg_data.loc[62500:87500],   # rest
        exg_data.loc[27500:32500],   # sw1
        exg_data.loc[62500:87500],   # sw2
        exg_data,                    # full session
    ]

    return [
        _galea_df_to_raw(seg.reset_index(drop=True))
        for seg in baseline_segments
    ]


def filter_data(raw: mne.io.RawArray) -> mne.io.RawArray:
    """Bandpass filter + notch (50 Hz & 60 Hz power line frequencies)."""
    # raw = raw.copy()
    # raw.filter(l_freq=0.5, h_freq=40.0, fir_design="firwin", verbose=False)
    # raw.notch_filter(freqs=[60.0, 120.0], verbose=False)

    # fig = raw.compute_psd(picks="eeg").plot(picks="eeg", show=False)
    # plt.show()
    # plt.clf()
    eeg_chans = mne.pick_types(raw.info,
                               eeg=True,
                               emg=False,
                               eog=False)

    freqs = []
    freq = LINE_FREQ
    while freq < raw.info['sfreq']/2:
        freqs.append(freq)
        freq += freq

    filtered_galea_mne = raw.copy().notch_filter(freqs=freqs, picks=eeg_chans)

    fig = filtered_galea_mne.compute_psd(picks="eeg").plot(picks="eeg", show=False)
    plt.show()
    plt.clf()

    eeg_chans = mne.pick_types(filtered_galea_mne.info,
                               eeg=True,
                               emg=False,
                               eog=False)

    lowcut = 0.5
    highcut = 100.0
    bandpass_data = filtered_galea_mne.copy().filter(l_freq=lowcut,
                                        h_freq=highcut,
                                        picks=eeg_chans)

    fig = bandpass_data.compute_psd(picks="eeg").plot(picks="eeg", show=False)
    plt.show()
    plt.clf()
    return bandpass_data


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

