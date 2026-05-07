import matplotlib.pyplot as plt
from load_preprocess import *
from overreliance import *
from system1_2 import *
from critical_thinking import *
from valence import *
from flow_state import *
from attention import *

BASELINE_FILE = "julia sandbox/galea_session_2026-05-05_17-09-55/recording_2026-05-05_17-10-02/openbci-raw-exg_2026-05-05_17-10-02.txt"
CH_NAMES = ["EOG 1", "EOG 2", "F1", "F2", "C3", "C4", "P3", "P4", "O1", "O2", "Cz", "Pz"]
SFREQ = 250  # Hz — change to 125 or 500 if you used a different rate

"""
will need to run all this on both the baseline and the actual task (which baseline? i think rest, probably), so can compare the differences
need to be able to look at one baseline - all 2 min, so can just filter for minutes 4-6
"""

if __name__ == "__main__":
    print(f"Loading: {BASELINE_FILE}")
    raw = load_galea_txt(BASELINE_FILE, CH_NAMES, SFREQ)
    print(f"  {raw.info['nchan']} channels, {raw.times[-1]:.1f} s @ {SFREQ} Hz")

    print("Preprocessing (0.5-40 Hz bandpass + 50, 60 Hz notch)...")
    raw_clean = filter_data(raw)

    print("removing eye blinks...")
    raw_noblink = remove_blinks(raw_clean)

    print("segmenting...")
    segs = segment_raw(raw_noblink)

    print("computing flow state features")
    print(compute_flow_features(raw_noblink))

    result = compute_ddtf_connectivity(raw_noblink)

    # Single band
    fig = plot_ddtf(result, band="alpha", threshold=0.25)
    fig.savefig("alpha_connectivity.png", dpi=150, bbox_inches="tight")

    # All three bands side-by-side (matches paper's figure layout)
    fig = plot_all_bands(result, threshold=0.25)
    fig.savefig("ddtf_all_bands.png", dpi=150, bbox_inches="tight")
    #plt.show()

    print("computing attention powers...")
    print(compute_attn_features(raw_noblink))

    result = compute_beta_alpha_ratio(raw_noblink)

    plt.clf()
    # Time-series of engagement over the recording
    plt.plot(result["times"], result["ratio_per_window"])
    plt.xlabel("Time (s)")
    plt.ylabel("β/α ratio")
    plt.title(f"Cognitive engagement  (mean = {result['mean_ratio']:.2f})")
    #plt.show()

    print("computing system 1/2 powers...")
    print(compute_sys12_features(raw_noblink))

    print("\nDone.")