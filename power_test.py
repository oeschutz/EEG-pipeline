from load_preprocess import *
from flow_state import *

BASELINE_FILE1 = "julia sandbox/galea_session_2026-05-05_17-09-55/recording_2026-05-05_17-10-02/openbci-raw-exg_2026-05-05_17-10-02.txt"
SFREQ=250
CH_NAMES = ["EOG 1", "EOG 2", "F1", "F2", "C3", "C4", "P3", "P4", "O1", "O2", "Cz", "Pz"]


print(f"Loading: {BASELINE_FILE1}")
baseline = load_galea_txt(BASELINE_FILE1, CH_NAMES, SFREQ) # load the data
print(f"  {baseline.info['nchan']} channels, {baseline.times[-1]:.1f} s @ {SFREQ} Hz")

print("Preprocessing (0.5-40 Hz bandpass + 50, 60 Hz notch)...")
baseline_clean = filter_data(baseline) # bandpass filter

print("removing eye blinks...")
baseline_noblink = remove_blinks(baseline_clean) # remove blink artifacts

# print("segmenting...")
# segs = segment_raw(baseline_noblink) # segment? the computations do their own segmenting so probably not necessary

baseline_noblink2 = mne.io.RawArray(
    baseline_noblink.get_data()[:, 250 : 1500],
    baseline_noblink.info,
    verbose=False,
)
print(len(baseline_noblink2))

print("computing flow state features for first 5 seconds after marker 8")
flow_fts_baseline = compute_flow_features(baseline_noblink2) # compute features associated with flow
baseline_noblink2.plot_psd(area_mode='range', tmax=10.0, show=False, average=True)
baseline_noblink2.compute_psd().plot()
plt.show()
print(flow_fts_baseline)