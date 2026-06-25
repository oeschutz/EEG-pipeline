import os
import matplotlib.pyplot as plt
from load_preprocess import *
from overreliance import *
from system1_2 import *
from critical_thinking import *
from valence import *
from flow_state import *
from attention import *

BASELINE_FILE1 = "C:/Users/olivi/Documents/Galea_GUI/Recordings/galea_session_p10-2026-06-18_15-43-32/recording_2026-06-18_16-06-55/openbci-raw-exg_2026-06-18_16-06-55.txt"
RAW_FILE1 = "C:/Users/olivi/Documents/Galea_GUI/Recordings/galea_session_p10-2026-06-18_15-43-32/recording_2026-06-18_16-16-21/openbci-raw-exg_2026-06-18_16-16-21.txt"
BASELINE_FILE2 = "C:/Users/olivi/Documents/Galea_GUI/Recordings/galea_session_p10-2026-06-18_15-43-32/recording_2026-06-18_16-55-15/openbci-raw-exg_2026-06-18_16-55-15.txt"
RAW_FILE2 = "C:/Users/olivi/Documents/Galea_GUI/Recordings/galea_session_p10-2026-06-18_15-43-32/recording_2026-06-18_17-04-06/openbci-raw-exg_2026-06-18_17-04-06.txt"
CH_NAMES = ["EOG 1", "EOG 2", "F1", "F2", "C3", "C4", "P3", "P4", "O1", "O2", "Cz", "Pz"]
SFREQ = 250  # Hz — change to 125 or 500 if you used a different rate

"""
will need to run all this on both the baseline and the actual task (which baseline? i think sw2, probably), so can compare the differences
need to be able to look at one baseline - all 2 min, so can just filter for minutes 4-6
"""

if __name__ == "__main__":
    # print(f"Loading: {BASELINE_FILE1}")
    # baseline = load_baseline_galea(BASELINE_FILE1, CH_NAMES, SFREQ) # load the data
    # math = baseline[0]
    # eyes = baseline[1]
    # rest = baseline[2]
    # sw1 = baseline[3]
    # sw2 = baseline[4]
    # full = baseline[5]
    # #print(f"  {baseline.info['nchan']} channels, {baseline.times[-1]:.1f} s @ {SFREQ} Hz")

    # print("Preprocessing (0.5-40 Hz bandpass + 50, 60 Hz notch)...")
    # math_clean = filter_data(math) # bandpass filter

    # print("removing eye blinks...")
    # math_noblink = remove_blinks(math_clean) # remove blink artifacts

    # # print("segmenting...")
    # # segs = segment_raw(baseline_noblink) # segment? the computations do their own segmenting so probably not necessary

    # print("computing flow state features for baseline")
    # flow_fts_math = compute_flow_features(math_noblink) # compute features associated with flow

    # ddtf_math = compute_ddtf_connectivity(math_noblink) # compute brain network connectivity using MIT paper methodology

    # # # Single band
    # fig = plot_ddtf(ddtf_math, band="all", threshold=0.25)
    # fig.savefig("all_connectivity_math.png", dpi=150, bbox_inches="tight")

    # # All three bands side-by-side (matches paper's figure layout)
    # fig = plot_all_bands(ddtf_math, threshold=0.25) # plot figure of connectivity in each band
    # fig.savefig("ddtf_all_bands_math.png", dpi=150, bbox_inches="tight")
    # #plt.show()

    # print("computing attention powers for baseline") 
    # print(compute_attn_features(math_noblink)) # compute features associated with attention

    # beta_alpha_math = compute_beta_alpha_ratio(math_noblink) # compute beta/alpha ratio (associated with cognitive engagement)

    # plt.clf()
    # # Time-series of engagement over the recording
    # plt.plot(beta_alpha_math["times"], beta_alpha_math["ratio_per_window"])
    # plt.xlabel("Time (s)")
    # plt.ylabel("β/α ratio")
    # plt.title(f"Cognitive engagement  (mean = {beta_alpha_math['mean_ratio']:.2f})") # graph beta/alpha ratio over time
    # plt.savefig("cognitive_engagement_math.png", dpi=150, bbox_inches="tight")
    # #plt.show()

    # print("computing system 1/2 powers for baseline")
    # sys12_fts_math = compute_sys12_features(math_noblink) # compute features associated with system 1/2 thinking

    # print("computing left frontal alpha for CT for baseline")
    # lfa_math = compute_left_frontal_alpha(math_noblink) # compute left frontal alpha power (associated with critical thinking)


    # print("Preprocessing (0.5-40 Hz bandpass + 50, 60 Hz notch)...")
    # eyes_clean = filter_data(eyes) # bandpass filter

    # print("removing eye blinks...")
    # eyes_noblink = remove_blinks(eyes_clean) # remove blink artifacts

    # # print("segmenting...")
    # # segs = segment_raw(baseline_noblink) # segment? the computations do their own segmenting so probably not necessary

    # print("computing flow state features for baseline")
    # flow_fts_eyes = compute_flow_features(eyes_noblink) # compute features associated with flow

    # ddtf_eyes = compute_ddtf_connectivity(eyes_noblink) # compute brain network connectivity using MIT paper methodology

    # # # Single band
    # fig = plot_ddtf(ddtf_eyes, band="all", threshold=0.25)
    # fig.savefig("all_connectivity_eyes.png", dpi=150, bbox_inches="tight")

    # # All three bands side-by-side (matches paper's figure layout)
    # fig = plot_all_bands(ddtf_eyes, threshold=0.25) # plot figure of connectivity in each band
    # fig.savefig("ddtf_all_bands_eyes.png", dpi=150, bbox_inches="tight")
    # #plt.show()

    # print("computing attention powers for baseline") 
    # print(compute_attn_features(eyes_noblink)) # compute features associated with attention

    # beta_alpha_eyes = compute_beta_alpha_ratio(eyes_noblink) # compute beta/alpha ratio (associated with cognitive engagement)

    # plt.clf()
    # # Time-series of engagement over the recording
    # plt.plot(beta_alpha_eyes["times"], beta_alpha_eyes["ratio_per_window"])
    # plt.xlabel("Time (s)")
    # plt.ylabel("β/α ratio")
    # plt.title(f"Cognitive engagement  (mean = {beta_alpha_eyes['mean_ratio']:.2f})") # graph beta/alpha ratio over time
    # plt.savefig("cognitive_engagement_eyes.png", dpi=150, bbox_inches="tight")
    # #plt.show()

    # print("computing system 1/2 powers for baseline")
    # sys12_fts_eyes = compute_sys12_features(eyes_noblink) # compute features associated with system 1/2 thinking

    # print("computing left frontal alpha for CT for baseline")
    # lfa_eyes = compute_left_frontal_alpha(eyes_noblink) # compute left frontal alpha power (associated with critical thinking)

    
    # print("Preprocessing (0.5-40 Hz bandpass + 50, 60 Hz notch)...")
    # rest_clean = filter_data(rest) # bandpass filter

    # print("removing eye blinks...")
    # rest_noblink = remove_blinks(rest_clean) # remove blink artifacts

    # # print("segmenting...")
    # # segs = segment_raw(baseline_noblink) # segment? the computations do their own segmenting so probably not necessary

    # print("computing flow state features for baseline")
    # flow_fts_rest = compute_flow_features(rest_noblink) # compute features associated with flow

    # ddtf_rest = compute_ddtf_connectivity(rest_noblink) # compute brain network connectivity using MIT paper methodology

    # # # Single band
    # fig = plot_ddtf(ddtf_rest, band="all", threshold=0.25)
    # fig.savefig("all_connectivity_rest.png", dpi=150, bbox_inches="tight")

    # # All three bands side-by-side (matches paper's figure layout)
    # fig = plot_all_bands(ddtf_rest, threshold=0.25) # plot figure of connectivity in each band
    # fig.savefig("ddtf_all_bands_rest.png", dpi=150, bbox_inches="tight")
    # #plt.show()

    # print("computing attention powers for baseline") 
    # print(compute_attn_features(rest_noblink)) # compute features associated with attention

    # beta_alpha_rest = compute_beta_alpha_ratio(rest_noblink) # compute beta/alpha ratio (associated with cognitive engagement)

    # plt.clf()
    # # Time-series of engagement over the recording
    # plt.plot(beta_alpha_rest["times"], beta_alpha_rest["ratio_per_window"])
    # plt.xlabel("Time (s)")
    # plt.ylabel("β/α ratio")
    # plt.title(f"Cognitive engagement  (mean = {beta_alpha_rest['mean_ratio']:.2f})") # graph beta/alpha ratio over time
    # plt.savefig("cognitive_engagement_rest.png", dpi=150, bbox_inches="tight")
    # #plt.show()

    # print("computing system 1/2 powers for baseline")
    # sys12_fts_rest = compute_sys12_features(rest_noblink) # compute features associated with system 1/2 thinking

    # print("computing left frontal alpha for CT for baseline")
    # lfa_rest = compute_left_frontal_alpha(rest_noblink) # compute left frontal alpha power (associated with critical thinking)


    # print("Preprocessing (0.5-40 Hz bandpass + 50, 60 Hz notch)...")
    # sw1_clean = filter_data(sw1) # bandpass filter

    # print("removing eye blinks...")
    # sw1_noblink = remove_blinks(sw1_clean) # remove blink artifacts

    # # print("segmenting...")
    # # segs = segment_raw(baseline_noblink) # segment? the computations do their own segmenting so probably not necessary

    # print("computing flow state features for baseline")
    # flow_fts_sw1 = compute_flow_features(sw1_noblink) # compute features associated with flow

    # ddtf_sw1 = compute_ddtf_connectivity(sw1_noblink) # compute brain network connectivity using MIT paper methodology

    # # # Single band
    # fig = plot_ddtf(ddtf_sw1, band="all", threshold=0.25)
    # fig.savefig("all_connectivity_sw1.png", dpi=150, bbox_inches="tight")

    # # All three bands side-by-side (matches paper's figure layout)
    # fig = plot_all_bands(ddtf_sw1, threshold=0.25) # plot figure of connectivity in each band
    # fig.savefig("ddtf_all_bands_sw1.png", dpi=150, bbox_inches="tight")
    # #plt.show()

    # print("computing attention powers for baseline") 
    # print(compute_attn_features(sw1_noblink)) # compute features associated with attention

    # beta_alpha_sw1 = compute_beta_alpha_ratio(sw1_noblink) # compute beta/alpha ratio (associated with cognitive engagement)

    # plt.clf()
    # # Time-series of engagement over the recording
    # plt.plot(beta_alpha_sw1["times"], beta_alpha_sw1["ratio_per_window"])
    # plt.xlabel("Time (s)")
    # plt.ylabel("β/α ratio")
    # plt.title(f"Cognitive engagement  (mean = {beta_alpha_sw1['mean_ratio']:.2f})") # graph beta/alpha ratio over time
    # plt.savefig("cognitive_engagement_sw1.png", dpi=150, bbox_inches="tight")
    # #plt.show()

    # print("computing system 1/2 powers for baseline")
    # sys12_fts_sw1 = compute_sys12_features(sw1_noblink) # compute features associated with system 1/2 thinking

    # print("computing left frontal alpha for CT for baseline")
    # lfa_sw1 = compute_left_frontal_alpha(sw1_noblink) # compute left frontal alpha power (associated with critical thinking)


    # print("Preprocessing (0.5-40 Hz bandpass + 50, 60 Hz notch)...")
    # sw2_clean = filter_data(sw2) # bandpass filter

    # print("removing eye blinks...")
    # sw2_noblink = remove_blinks(sw2_clean) # remove blink artifacts

    # # print("segmenting...")
    # # segs = segment_raw(baseline_noblink) # segment? the computations do their own segmenting so probably not necessary

    # print("computing flow state features for baseline")
    # flow_fts_sw2 = compute_flow_features(sw2_noblink) # compute features associated with flow

    # ddtf_sw2 = compute_ddtf_connectivity(sw2_noblink) # compute brain network connectivity using MIT paper methodology

    # # # Single band
    # fig = plot_ddtf(ddtf_sw2, band="all", threshold=0.25)
    # fig.savefig("all_connectivity_sw2.png", dpi=150, bbox_inches="tight")

    # # All three bands side-by-side (matches paper's figure layout)
    # fig = plot_all_bands(ddtf_sw2, threshold=0.25) # plot figure of connectivity in each band
    # fig.savefig("ddtf_all_bands_sw2.png", dpi=150, bbox_inches="tight")
    # #plt.show()

    # print("computing attention powers for baseline") 
    # print(compute_attn_features(sw2_noblink)) # compute features associated with attention

    # beta_alpha_sw2 = compute_beta_alpha_ratio(sw2_noblink) # compute beta/alpha ratio (associated with cognitive engagement)

    # plt.clf()
    # # Time-series of engagement over the recording
    # plt.plot(beta_alpha_sw2["times"], beta_alpha_sw2["ratio_per_window"])
    # plt.xlabel("Time (s)")
    # plt.ylabel("β/α ratio")
    # plt.title(f"Cognitive engagement  (mean = {beta_alpha_sw2['mean_ratio']:.2f})") # graph beta/alpha ratio over time
    # plt.savefig("cognitive_engagement_sw2.png", dpi=150, bbox_inches="tight")
    # #plt.show()

    # print("computing system 1/2 powers for baseline")
    # sys12_fts_sw2 = compute_sys12_features(sw2_noblink) # compute features associated with system 1/2 thinking

    # print("computing left frontal alpha for CT for baseline")
    # lfa_sw2 = compute_left_frontal_alpha(sw2_noblink) # compute left frontal alpha power (associated with critical thinking)

    # print("Preprocessing (0.5-40 Hz bandpass + 50, 60 Hz notch)...")
    # full_clean = filter_data(full) # bandpass filter

    # print("removing eye blinks...")
    # full_noblink = remove_blinks(full_clean) # remove blink artifacts

    # # print("segmenting...")
    # # segs = segment_raw(baseline_noblink) # segment? the computations do their own segmenting so probably not necessary

    # print("computing flow state features for baseline")
    # flow_fts_full = compute_flow_features(full_noblink) # compute features associated with flow

    # ddtf_full = compute_ddtf_connectivity(full_noblink) # compute brain network connectivity using MIT paper methodology

    # # # Single band
    # fig = plot_ddtf(ddtf_full, band="all", threshold=0.25)
    # fig.savefig("all_connectivity_full.png", dpi=150, bbox_inches="tight")

    # # All three bands side-by-side (matches paper's figure layout)
    # fig = plot_all_bands(ddtf_full, threshold=0.25) # plot figure of connectivity in each band
    # fig.savefig("ddtf_all_bands_full.png", dpi=150, bbox_inches="tight")
    # #plt.show()

    # print("computing attention powers for baseline") 
    # print(compute_attn_features(full_noblink)) # compute features associated with attention

    # beta_alpha_full = compute_beta_alpha_ratio(full_noblink) # compute beta/alpha ratio (associated with cognitive engagement)

    # plt.clf()
    # # Time-series of engagement over the recording
    # plt.plot(beta_alpha_full["times"], beta_alpha_full["ratio_per_window"])
    # plt.xlabel("Time (s)")
    # plt.ylabel("β/α ratio")
    # plt.title(f"Cognitive engagement  (mean = {beta_alpha_full['mean_ratio']:.2f})") # graph beta/alpha ratio over time
    # plt.savefig("cognitive_engagement_full.png", dpi=150, bbox_inches="tight")
    # #plt.show()

    # print("computing system 1/2 powers for baseline")
    # sys12_fts_full = compute_sys12_features(full_noblink) # compute features associated with system 1/2 thinking

    # print("computing left frontal alpha for CT for baseline")
    # lfa_full = compute_left_frontal_alpha(full_noblink) # compute left frontal alpha power (associated with critical thinking)


    # #####################################################
    # ###                   actual task                 ###
    # #####################################################


    # print(f"Loading: {RAW_FILE1}")
    # raw = load_galea_txt(RAW_FILE1, CH_NAMES, SFREQ) # load the data
    # print(f"  {raw.info['nchan']} channels, {raw.times[-1]:.1f} s @ {SFREQ} Hz")

    # print("Preprocessing (0.5-40 Hz bandpass + 50, 60 Hz notch)...")
    # raw_clean = filter_data(raw) # bandpass filter

    # print("removing eye blinks...")
    # raw_noblink = remove_blinks(raw_clean) # remove blink artifacts

    # # print("segmenting...")
    # # segs = segment_raw(raw_noblink) # segment? the computations do their own segmenting so probably not necessary

    # print("computing flow state features for raw")
    # flow_fts_raw = compute_flow_features(raw_noblink) # compute features associated with flow

    # ddtf_raw = compute_ddtf_connectivity(raw_noblink) # compute brain network connectivity using MIT paper methodology

    # # # Single band
    # fig = plot_ddtf(ddtf_raw, band="all", threshold=0.25)
    # fig.savefig("all_connectivity_raw.png", dpi=150, bbox_inches="tight")

    # # All three bands side-by-side
    # fig = plot_all_bands(ddtf_raw, threshold=0.25) # plot figure of connectivity in each band
    # fig.savefig("ddtf_all_bands_raw.png", dpi=150, bbox_inches="tight")
    # #plt.show()

    # print("computing attention powers for raw") 
    # print(compute_attn_features(raw_noblink)) # compute features associated with attention

    # beta_alpha_raw = compute_beta_alpha_ratio(raw_noblink) # compute beta/alpha ratio (associated with cognitive engagement)

    # plt.clf()
    # # Time-series of engagement over the recording
    # plt.plot(beta_alpha_raw["times"], beta_alpha_raw["ratio_per_window"])
    # plt.xlabel("Time (s)")
    # plt.ylabel("β/α ratio")
    # plt.title(f"Cognitive engagement  (mean = {beta_alpha_raw['mean_ratio']:.2f})") # graph beta/alpha ratio over time
    # plt.savefig("cognitive_engagement_raw.png", dpi=150, bbox_inches="tight")
    # #plt.show()

    # print("computing system 1/2 powers for raw")
    # sys12_fts_raw = compute_sys12_features(raw_noblink) # compute features associated with system 1/2 thinking

    # print("computing left frontal alpha for CT for raw")
    # lfa_raw = compute_left_frontal_alpha(raw_noblink) # compute left frontal alpha power (associated with critical thinking)



    #####################################################
    ###                   comparison                  ###
    #####################################################
    from comparison import (
        compare_flow_state, compare_cognitive_load,
        compare_attention, compare_system12, compare_valence,
        print_comparison_report,
    )
    from temporal_analysis import (
        run_temporal_analysis, print_temporal_report,
        print_temporal_comparison_report,
    )

    SESSIONS = [
        ("session1", BASELINE_FILE1, RAW_FILE1),
        ("session2", BASELINE_FILE2, RAW_FILE2),
    ]

    def run_session_analysis(session_name: str, baseline_file: str, task_file: str,
                             output_dir: str):
        """Run full baseline-vs-task comparison and temporal analysis for one pair."""
        os.makedirs(output_dir, exist_ok=True)
        snapshots_dir = os.path.join(output_dir, "state_snapshots")

        print(f"\n{'=' * 72}")
        print(f"  {session_name.upper()}")
        print(f"{'=' * 72}")
        print(f"  Baseline : {baseline_file}")
        print(f"  Task     : {task_file}")
        print(f"  Output   : {output_dir}/")

        baseline_raw = filter_data(load_baseline_galea(baseline_file, CH_NAMES, SFREQ)[4])
        baseline_raw = remove_blinks(baseline_raw)           # use sw2 as resting baseline
        task_raw     = filter_data(load_galea_txt(task_file, CH_NAMES, SFREQ))
        task_raw     = remove_blinks(task_raw)

        bl_flow  = compute_flow_features(baseline_raw)
        bl_sys12 = compute_sys12_features(baseline_raw)
        bl_attn  = compute_attn_features(baseline_raw)
        bl_ddtf  = compute_ddtf_connectivity(baseline_raw)
        fig = plot_ddtf(bl_ddtf, band="all", threshold=0.25)
        fig.savefig(os.path.join(output_dir, "baseline_all.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

        fig = plot_all_bands(bl_ddtf, threshold=0.25)
        fig.savefig(os.path.join(output_dir, "baseline_side_by_side.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

        bl_val = compute_beta_alpha_ratio(baseline_raw)

        tk_flow  = compute_flow_features(task_raw)
        tk_sys12 = compute_sys12_features(task_raw)
        tk_attn  = compute_attn_features(task_raw)
        tk_ddtf  = compute_ddtf_connectivity(task_raw)
        fig = plot_ddtf(tk_ddtf, band="all", threshold=0.25)
        fig.savefig(os.path.join(output_dir, "task_all.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

        fig = plot_all_bands(tk_ddtf, threshold=0.25)
        fig.savefig(os.path.join(output_dir, "task_side_by_side.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

        tk_val = compute_beta_alpha_ratio(task_raw)

        results = {
            "flow_state"     : compare_flow_state(bl_flow, tk_flow),
            "cognitive_load" : compare_cognitive_load(bl_sys12, tk_sys12),
            "attention"      : compare_attention(bl_attn, tk_attn, bl_ddtf, tk_ddtf),
            "system_1_2"     : compare_system12(bl_flow, tk_flow, bl_sys12, tk_sys12),
            "valence"        : compare_valence(bl_val, tk_val),
        }
        print_comparison_report(results)

        print(f"\nRunning temporal state detection for {session_name}...")
        temporal = run_temporal_analysis(
            task_raw,
            baseline_flow=bl_flow,
            baseline_sys12=bl_sys12,
            baseline_attn=bl_attn,
            baseline_val=bl_val,
            task_val=tk_val,
            output_dir=snapshots_dir,
        )
        report_path = os.path.join(output_dir, "temporal_report.txt")
        saved = print_temporal_report(temporal, output_path=report_path)
        if saved:
            print(f"  Temporal report saved -> {saved}")
        return temporal

    temporal_results = {}
    for session_name, baseline_file, task_file in SESSIONS:
        temporal_results[session_name] = run_session_analysis(
            session_name, baseline_file, task_file,
            output_dir=os.path.join("results", session_name),
        )

    comparison_path = os.path.join("results", "temporal_comparison.txt")
    saved = print_temporal_comparison_report(
        temporal_results["session1"],
        temporal_results["session2"],
        label_a="Task 1",
        label_b="Task 2",
        output_path=comparison_path,
    )
    if saved:
        print(f"  Temporal comparison saved -> {saved}")

    print("\nDone.")