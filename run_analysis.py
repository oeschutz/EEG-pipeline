import matplotlib.pyplot as plt
from load_preprocess import *
from overreliance import *
from system1_2 import *
from critical_thinking import *
from valence import *
from flow_state import *
from attention import *

BASELINE_FILE = "julia sandbox/galea_session_2026-05-05_17-09-55/recording_2026-05-05_17-10-02/openbci-raw-exg_2026-05-05_17-10-02.txt"
RAW_FILE = 'julia sandbox/galea_session_2026-05-05_17-09-55/recording_2026-05-05_17-24-22/openbci-raw-exg_2026-05-05_17-24-22.txt'
CH_NAMES = ["EOG 1", "EOG 2", "F1", "F2", "C3", "C4", "P3", "P4", "O1", "O2", "Cz", "Pz"]
SFREQ = 250  # Hz — change to 125 or 500 if you used a different rate

"""
will need to run all this on both the baseline and the actual task (which baseline? i think rest, probably), so can compare the differences
need to be able to look at one baseline - all 2 min, so can just filter for minutes 4-6
"""

if __name__ == "__main__":
    print(f"Loading: {BASELINE_FILE}")
    baseline = load_galea_txt(BASELINE_FILE, CH_NAMES, SFREQ) # load the data
    print(f"  {baseline.info['nchan']} channels, {baseline.times[-1]:.1f} s @ {SFREQ} Hz")

    print("Preprocessing (0.5-40 Hz bandpass + 50, 60 Hz notch)...")
    baseline_clean = filter_data(baseline) # bandpass filter

    print("removing eye blinks...")
    baseline_noblink = remove_blinks(baseline_clean) # remove blink artifacts

    # print("segmenting...")
    # segs = segment_raw(baseline_noblink) # segment? the computations do their own segmenting so probably not necessary

    print("computing flow state features for baseline")
    flow_fts_baseline = compute_flow_features(baseline_noblink) # compute features associated with flow

    ddtf_baseline = compute_ddtf_connectivity(baseline_noblink) # compute brain network connectivity using MIT paper methodology

    # # Single band
    # fig = plot_ddtf(result, band="alpha", threshold=0.25)
    # fig.savefig("alpha_connectivity.png", dpi=150, bbox_inches="tight")

    # All three bands side-by-side (matches paper's figure layout)
    fig = plot_all_bands(ddtf_baseline, threshold=0.25) # plot figure of connectivity in each band
    fig.savefig("ddtf_all_bands_baseline.png", dpi=150, bbox_inches="tight")
    #plt.show()

    print("computing attention powers for baseline") 
    print(compute_attn_features(baseline_noblink)) # compute features associated with attention

    beta_alpha_baseline = compute_beta_alpha_ratio(baseline_noblink) # compute beta/alpha ratio (associated with cognitive engagement)

    plt.clf()
    # Time-series of engagement over the recording
    plt.plot(beta_alpha_baseline["times"], beta_alpha_baseline["ratio_per_window"])
    plt.xlabel("Time (s)")
    plt.ylabel("β/α ratio")
    plt.title(f"Cognitive engagement  (mean = {beta_alpha_baseline['mean_ratio']:.2f})") # graph beta/alpha ratio over time
    plt.savefig("cognitive_engagement_baseline.png", dpi=150, bbox_inches="tight")
    #plt.show()

    print("computing system 1/2 powers for baseline")
    sys12_fts_baseline = compute_sys12_features(baseline_noblink) # compute features associated with system 1/2 thinking

    print("computing left frontal alpha for CT for baseline")
    lfa_baseline = compute_left_frontal_alpha(baseline_noblink) # compute left frontal alpha power (associated with critical thinking)

    #####################################################
    ###                   actual task                 ###
    #####################################################


    print(f"Loading: {RAW_FILE}")
    raw = load_galea_txt(RAW_FILE, CH_NAMES, SFREQ) # load the data
    print(f"  {raw.info['nchan']} channels, {raw.times[-1]:.1f} s @ {SFREQ} Hz")

    print("Preprocessing (0.5-40 Hz bandpass + 50, 60 Hz notch)...")
    raw_clean = filter_data(raw) # bandpass filter

    print("removing eye blinks...")
    raw_noblink = remove_blinks(raw_clean) # remove blink artifacts

    # print("segmenting...")
    # segs = segment_raw(raw_noblink) # segment? the computations do their own segmenting so probably not necessary

    print("computing flow state features for raw")
    flow_fts_raw = compute_flow_features(raw_noblink) # compute features associated with flow

    ddtf_raw = compute_ddtf_connectivity(raw_noblink) # compute brain network connectivity using MIT paper methodology

    # # Single band
    # fig = plot_ddtf(result, band="alpha", threshold=0.25)
    # fig.savefig("alpha_connectivity.png", dpi=150, bbox_inches="tight")

    # All three bands side-by-side (matches paper's figure layout)
    fig = plot_all_bands(ddtf_raw, threshold=0.25) # plot figure of connectivity in each band
    fig.savefig("ddtf_all_bands_raw.png", dpi=150, bbox_inches="tight")
    #plt.show()

    print("computing attention powers for raw") 
    print(compute_attn_features(raw_noblink)) # compute features associated with attention

    beta_alpha_raw = compute_beta_alpha_ratio(raw_noblink) # compute beta/alpha ratio (associated with cognitive engagement)

    plt.clf()
    # Time-series of engagement over the recording
    plt.plot(beta_alpha_raw["times"], beta_alpha_raw["ratio_per_window"])
    plt.xlabel("Time (s)")
    plt.ylabel("β/α ratio")
    plt.title(f"Cognitive engagement  (mean = {beta_alpha_raw['mean_ratio']:.2f})") # graph beta/alpha ratio over time
    plt.savefig("cognitive_engagement_raw.png", dpi=150, bbox_inches="tight")
    #plt.show()

    print("computing system 1/2 powers for raw")
    sys12_fts_raw = compute_sys12_features(raw_noblink) # compute features associated with system 1/2 thinking

    print("computing left frontal alpha for CT for raw")
    lfa_raw = compute_left_frontal_alpha(raw_noblink) # compute left frontal alpha power (associated with critical thinking)

    #####################################################
    ###                   comparison                  ###
    #####################################################

    print(beta_alpha_raw)

    print("\nDone.")