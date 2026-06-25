import mne
import numpy as np
from typing import TypedDict
from scipy.linalg import solve_triangular
from scipy.signal import csd


# ── Channel layout ─────────────────────────────────────────────────────────────
EEG_CHANNELS = ["F1", "F2", "C3", "C4", "P3", "P4", "O1", "O2", "Cz", "Pz"]

# ── Frequency bands (Hz) ───────────────────────────────────────────────────────
DELTA_BAND = (0.5, 4.0)
ALPHA_BAND = (8.0, 13.0)
BETA_BAND  = (13.0, 30.0)
ALL_BANDS = (0.5, 40.0)


class dDTFResult(TypedDict):
    alpha: np.ndarray   # (n_ch, n_ch) mean dDTF in alpha band; [i,j] = j→i
    beta:  np.ndarray   # (n_ch, n_ch) mean dDTF in beta  band
    delta: np.ndarray   # (n_ch, n_ch) mean dDTF in delta band
    all:   np.ndarray
    ch_names: list[str] # channel labels corresponding to matrix axes


# ── 1. MVAR model fitting ──────────────────────────────────────────────────────

def _fit_mvar(data: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit a Multivariate AutoRegressive (MVAR) model of a given order to
    multi-channel data using ordinary least squares (Yule-Walker / OLS).

    The MVAR model is:
        X(t) = A(1)·X(t-1) + A(2)·X(t-2) + ... + A(p)·X(t-p) + E(t)

    This is solved in one shot by stacking the lagged design matrix and
    solving the normal equations, equivalent to multivariate OLS.

    Parameters
    ----------
    data  : (n_channels, n_times) array
    order : model order p (number of lags)

    Returns
    -------
    A_coeffs : (n_channels, n_channels * order) coefficient matrix.
               Reshape to (order, n_ch, n_ch) to get A(1)…A(p).
    noise_cov : (n_channels, n_channels) residual noise covariance matrix.
    """
    n_ch, n_t = data.shape

    # Build the lagged design matrix: each row is [X(t-1)ᵀ, …, X(t-p)ᵀ]
    n_obs = n_t - order
    X_lagged = np.zeros((n_obs, n_ch * order))
    for lag in range(1, order + 1):
        col_start = (lag - 1) * n_ch
        X_lagged[:, col_start : col_start + n_ch] = data[:, order - lag : n_t - lag].T

    Y = data[:, order:].T                         # (n_obs, n_ch)

    # OLS: A = (XᵀX)⁻¹ Xᵀ Y  →  solved via lstsq for numerical stability
    A_coeffs, residuals, _, _ = np.linalg.lstsq(X_lagged, Y, rcond=None)
    A_coeffs = A_coeffs.T                         # (n_ch, n_ch * order)

    E = Y - X_lagged @ A_coeffs.T                 # (n_obs, n_ch) residuals
    noise_cov = (E.T @ E) / (n_obs - n_ch * order - 1)

    return A_coeffs, noise_cov


def _select_mvar_order(data: np.ndarray,
                       max_order: int = 30) -> int:
    """
    Select optimal MVAR model order using the Schwarz Bayesian Criterion (SBC /
    BIC), as recommended by the dDTF literature.

    SBC(p) = ln|Σ_p| + (n_ch² · p · ln(N)) / N

    Parameters
    ----------
    data      : (n_channels, n_times)
    max_order : upper bound on search range

    Returns
    -------
    Optimal model order (int).
    """
    n_ch, n_t = data.shape
    best_sbc   = np.inf
    best_order = 1

    for p in range(1, max_order + 1):
        _, noise_cov = _fit_mvar(data, p)
        sign, log_det = np.linalg.slogdet(noise_cov)
        if sign <= 0:
            continue
        sbc = log_det + (n_ch ** 2 * p * np.log(n_t)) / n_t
        if sbc < best_sbc:
            best_sbc   = sbc
            best_order = p

    return best_order


# ── 2. Transfer matrix H(f) ────────────────────────────────────────────────────

def _transfer_matrix(A_coeffs: np.ndarray,
                     n_ch: int,
                     order: int,
                     freqs: np.ndarray,
                     sfreq: float) -> np.ndarray:
    """
    Compute the MVAR transfer matrix H(f) at each frequency.

    H(f) = A(f)⁻¹  where  A(f) = I - Σₖ A(k)·e^{-j2πfk/sfreq}

    Parameters
    ----------
    A_coeffs : (n_ch, n_ch * order) MVAR coefficient matrix
    freqs    : 1-D array of frequencies at which to evaluate H(f)  [Hz]
    sfreq    : sampling frequency  [Hz]

    Returns
    -------
    H : (n_freqs, n_ch, n_ch) complex transfer matrix
    """
    n_freqs = len(freqs)
    H = np.zeros((n_freqs, n_ch, n_ch), dtype=complex)

    # Reshape to (order, n_ch, n_ch): A_coeffs[:, k*n_ch:(k+1)*n_ch] = A(k+1)
    A_lags = A_coeffs.reshape(n_ch, order, n_ch).transpose(1, 0, 2)  # (p, n_ch, n_ch)

    for fi, f in enumerate(freqs):
        # A(f) = I - Σ_{k=1}^{p} A(k) · e^{-j2πfk/sfreq}
        A_f = np.eye(n_ch, dtype=complex)
        for k in range(order):
            phase = np.exp(-1j * 2 * np.pi * f * (k + 1) / sfreq)
            A_f -= A_lags[k] * phase

        H[fi] = np.linalg.inv(A_f)

    return H


# ── 3. Partial coherence ───────────────────────────────────────────────────────

def _partial_coherence(H: np.ndarray,
                       noise_cov: np.ndarray) -> np.ndarray:
    """
    Compute partial coherence C_ij(f) from the transfer matrix and noise
    covariance, following Korzeniewska et al. (2003).

    The cross-spectral matrix is  S(f) = H(f) · Σ_E · H(f)^H
    Partial coherence is derived from the inverse of S(f):
        C_ij(f) = -M_ij(f) / sqrt(M_ii(f) · M_jj(f))
    where M(f) = S(f)⁻¹.

    Parameters
    ----------
    H         : (n_freqs, n_ch, n_ch) transfer matrix
    noise_cov : (n_ch, n_ch) noise covariance matrix

    Returns
    -------
    partial_coh : (n_freqs, n_ch, n_ch) partial coherence magnitudes ∈ [0, 1]
    """
    n_freqs, n_ch, _ = H.shape
    partial_coh = np.zeros((n_freqs, n_ch, n_ch))

    for fi in range(n_freqs):
        S = H[fi] @ noise_cov @ H[fi].conj().T     # cross-spectral matrix
        M = np.linalg.inv(S)                        # inverse spectral matrix

        for i in range(n_ch):
            for j in range(n_ch):
                denom = np.sqrt(np.abs(M[i, i]) * np.abs(M[j, j]))
                if denom > 0:
                    partial_coh[fi, i, j] = np.abs(M[i, j]) / denom

    return partial_coh


# ── 4. dDTF ───────────────────────────────────────────────────────────────────

def _compute_ddtf(H: np.ndarray,
                  noise_cov: np.ndarray) -> np.ndarray:
    """
    Compute the direct Directed Transfer Function (dDTF).

    dDTF_j→i(f) = ffDTF_j→i(f) · C_ij(f)

    where ffDTF uses a frequency-independent denominator (full-frequency DTF):
        ffDTF²_j→i(f) = |H_ij(f)|² / Σ_f Σ_m |H_im(f)|²

    Multiplying by partial coherence suppresses indirect (cascade) connections,
    leaving only direct directed flows.

    Parameters
    ----------
    H         : (n_freqs, n_ch, n_ch) transfer matrix
    noise_cov : (n_ch, n_ch) noise covariance

    Returns
    -------
    ddtf : (n_freqs, n_ch, n_ch) dDTF values, row i ← column j
    """
    n_freqs, n_ch, _ = H.shape

    # ── ffDTF: denominator is summed over ALL frequencies and input channels
    H_sq = np.abs(H) ** 2                                # (n_freqs, n_ch, n_ch)
    denom = H_sq.sum(axis=(0, 2), keepdims=True)         # (1, n_ch, 1) sum over f & inputs
    denom = np.where(denom == 0, 1e-12, denom)
    ff_dtf = np.sqrt(H_sq / denom)                       # (n_freqs, n_ch, n_ch)

    # ── Partial coherence
    partial_coh = _partial_coherence(H, noise_cov)       # (n_freqs, n_ch, n_ch)

    # ── dDTF = ffDTF × partial coherence
    ddtf = ff_dtf * partial_coh                          # (n_freqs, n_ch, n_ch)

    return ddtf


# ── 5. Band averaging ─────────────────────────────────────────────────────────

def _band_mean(ddtf: np.ndarray,
               freqs: np.ndarray,
               fmin: float,
               fmax: float) -> np.ndarray:
    """Average dDTF matrix over frequency bins within [fmin, fmax)."""
    mask = (freqs >= fmin) & (freqs < fmax)
    if not mask.any():
        raise ValueError(f"No frequency bins in [{fmin}, {fmax}) Hz.")
    return ddtf[mask].mean(axis=0)


# ── 6. Public entry point ─────────────────────────────────────────────────────

def compute_ddtf_connectivity(raw: mne.io.RawArray,
                               mvar_order: int | None = None,
                               freq_resolution: float = 0.5) -> dDTFResult:
    """
    Compute band-averaged dDTF directed connectivity from a raw EEG recording.

    The dDTF (direct Directed Transfer Function) is a measure of *effective*
    connectivity — it captures directed, frequency-specific information flow
    between channels while suppressing spurious cascade paths via partial
    coherence weighting.

    Pipeline
    --------
    1. Extract EEG data (EOG channels dropped).
    2. Select MVAR model order via SBC / BIC (or use caller-supplied value).
    3. Fit an MVAR model using multivariate OLS.
    4. Compute the transfer matrix H(f) via spectral factorisation.
    5. Compute partial coherence from the inverse cross-spectral matrix.
    6. Form dDTF = ffDTF × partial coherence at each frequency bin.
    7. Average within delta, alpha, and beta bands.

    Parameters
    ----------
    raw             : mne.io.RawArray
        Cleaned, bandpass-filtered (0.5–40 Hz) continuous EEG.
    mvar_order      : int or None
        MVAR model order p. If None (default), selected automatically by SBC.
        Values of 5–20 are typical for resting EEG at 128–512 Hz.
    freq_resolution : float
        Spacing between frequency evaluation points in Hz (default 0.5 Hz).
        Finer resolution is more precise but slower.

    Returns
    -------
    dDTFResult
        Dictionary with keys 'alpha', 'beta', 'delta', each an (n_ch × n_ch)
        connectivity matrix, and 'ch_names'.

        Matrix convention: result['alpha'][i, j] is the directed flow from
        channel j → channel i in the alpha band.
    """
    # ── Extract EEG-only data ─────────────────────────────────────────────────
    raw_eeg = raw.copy().pick_channels(EEG_CHANNELS)
    data    = raw_eeg.get_data()            # (n_ch, n_times), µV
    sfreq   = raw_eeg.info["sfreq"]
    n_ch    = data.shape[0]

    # Z-score each channel: MVAR fitting is sensitive to scale differences
    data = (data - data.mean(axis=1, keepdims=True)) / (
        data.std(axis=1, keepdims=True) + 1e-12
    )

    # ── Select MVAR order ─────────────────────────────────────────────────────
    if mvar_order is None:
        # Limit search to a sensible range; SBC search over long data is fast
        max_order  = min(30, int(sfreq // 4))
        mvar_order = _select_mvar_order(data, max_order=max_order)

    # ── Fit MVAR ──────────────────────────────────────────────────────────────
    A_coeffs, noise_cov = _fit_mvar(data, mvar_order)

    # ── Build frequency axis ──────────────────────────────────────────────────
    nyq   = sfreq / 2.0
    freqs = np.arange(freq_resolution, nyq, freq_resolution)  # exclude 0 Hz

    # ── Transfer matrix & dDTF ────────────────────────────────────────────────
    H    = _transfer_matrix(A_coeffs, n_ch, mvar_order, freqs, sfreq)
    ddtf = _compute_ddtf(H, noise_cov)

    # ── Band averages ─────────────────────────────────────────────────────────
    return dDTFResult(
        delta    = _band_mean(ddtf, freqs, *DELTA_BAND),
        alpha    = _band_mean(ddtf, freqs, *ALPHA_BAND),
        beta     = _band_mean(ddtf, freqs, *BETA_BAND),
        all      = _band_mean(ddtf, freqs, *ALL_BANDS),
        ch_names = raw_eeg.ch_names,
    )

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


# ── Scalp positions in normalized [0,1]x[0,1] space (10-20 layout) ────────────
SCALP_POS = {
    "F1":  (-0.18,  0.50), "F2":  ( 0.18,  0.50),
    "C3":  (-0.42,  0.00), "C4":  ( 0.42,  0.00),
    "P3":  (-0.28, -0.45), "P4":  ( 0.28, -0.45),
    "O1":  (-0.18, -0.75), "O2":  ( 0.18, -0.75),
    "Cz":  ( 0.00,  0.22), "Pz":  ( 0.00, -0.28),
}

BAND_CMAPS = {
    "alpha": "Blues",
    "beta":  "Greens",
    "delta": "Purples",
    "all": "Reds",
}

BAND_RANGES = {
    "alpha": "8–13 Hz",
    "beta":  "13–30 Hz",
    "delta": "0.5–4 Hz",
    "all": "0.5-40 Hz",
}


def _draw_arc(ax, p1, p2, strength, color, max_lw=4.0, bow=0.25):
    """Draw a curved directed arrow from p1 to p2 scaled by `strength` ∈ [0,1]."""
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    # Control point perpendicular to the midpoint
    mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
    length = np.sqrt(dx**2 + dy**2) + 1e-9
    nx, ny = -dy / length, dx / length
    cp = (mx + nx * bow, my + ny * bow)

    lw    = 0.3 + strength * max_lw
    alpha = 0.25 + strength * 0.75

    # FancyArrowPatch via connectionstyle quad bezier
    arrow = FancyArrowPatch(
        posA=p1, posB=p2,
        arrowstyle="-|>",
        connectionstyle=f"arc3,rad={bow * 0.8:.2f}",
        linewidth=lw,
        color=color,
        alpha=alpha,
        mutation_scale=6 + strength * 8,
        zorder=2,
    )
    ax.add_patch(arrow)


def plot_ddtf(
    result: dict,
    band: str = "alpha",
    threshold: float = 0.20,
    figsize: tuple = (7, 7),
    title: str | None = None,
) -> plt.Figure:
    """
    Render a circular scalp-topography connectivity diagram matching the
    paper's dDTF visualizations.

    Parameters
    ----------
    result    : dDTFResult dict from compute_ddtf_connectivity()
    band      : one of 'alpha', 'beta', 'delta', 'all'
    threshold : minimum normalized strength to display (0–1). Connections
                below this value are hidden to reduce visual clutter.
    figsize   : matplotlib figure size in inches
    title     : optional subplot title; defaults to band name + Hz range

    Returns
    -------
    matplotlib Figure
    """
    mat      = result[band]                   # (n_ch, n_ch)
    ch_names = result["ch_names"]
    n_ch     = len(ch_names)

    # Normalise to [0, 1] relative to the strongest connection
    max_val  = mat.max()
    norm_mat = mat / max_val if max_val > 0 else mat

    cmap = plt.get_cmap(BAND_CMAPS[band])

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    ax.set_aspect("equal")
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.05, 1.15)
    ax.axis("off")

    # ── Head outline ────────────────────────────────────────────────────────
    head = plt.Circle((0, 0), radius=0.92,
                      fill=False, linewidth=1.2, color="#CCCCCC", zorder=1)
    ax.add_patch(head)
    # Nose
    nose_x = [-.10, 0, .10]
    nose_y = [0.92, 1.05, 0.92]
    ax.plot(nose_x, nose_y, color="#CCCCCC", linewidth=1.2, zorder=1)
    # Ears
    for side in (-1, 1):
        ear = mpatches.Arc((side * 0.92, 0), 0.10, 0.22,
                           angle=0, theta1=270, theta2=90,
                           color="#CCCCCC", linewidth=1.2, zorder=1)
        ax.add_patch(ear)

    # ── Map channel names → 2-D coordinates ─────────────────────────────────
    coords = {}
    for i, ch in enumerate(ch_names):
        if ch in SCALP_POS:
            x, y = SCALP_POS[ch]
        else:
            # Fallback: evenly space unknowns around a circle
            angle = 2 * np.pi * i / n_ch - np.pi / 2
            x, y  = 0.7 * np.cos(angle), 0.7 * np.sin(angle)
        coords[ch] = (x, y)

    # ── Draw arcs ────────────────────────────────────────────────────────────
    for i, frm in enumerate(ch_names):
        for j, to in enumerate(ch_names):
            if i == j:
                continue
            strength = norm_mat[j, i]     # [j,i] = flow from i→j
            if strength < threshold:
                continue
            color = cmap(0.35 + strength * 0.65)
            _draw_arc(ax, coords[frm], coords[to], strength, color)

    # ── Draw electrode nodes ─────────────────────────────────────────────────
    # Node radius scales with total outflow strength
    out_strength = norm_mat.sum(axis=0)   # sum over targets for each source
    out_strength = out_strength / (out_strength.max() + 1e-9)

    for ch in ch_names:
        x, y  = coords[ch]
        r     = 0.045 + out_strength[ch_names.index(ch)] * 0.035
        color = cmap(0.6)

        circle = plt.Circle((x, y), radius=r, color=color,
                             zorder=5, linewidth=1.5,
                             ec="white")
        ax.add_patch(circle)
        ax.text(x, y, ch, ha="center", va="center",
                fontsize=8, fontweight="bold", color="white", zorder=6)

    # ── Colorbar legend ──────────────────────────────────────────────────────
    sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=max_val))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal",
                        fraction=0.035, pad=0.02, aspect=30)
    cbar.set_label("dDTF connectivity strength", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # ── Title ────────────────────────────────────────────────────────────────
    band_label = title or f"{band.capitalize()} band  ({BAND_RANGES[band]})"
    ax.set_title(band_label, fontsize=12, fontweight="bold", pad=10)

    fig.tight_layout()
    return fig


def plot_all_bands(
    result: dict,
    threshold: float = 0.20,
    suptitle: str | None = None,
) -> plt.Figure:
    """
    Render alpha, beta, delta, and full-band connectivity in a single figure.

    Parameters
    ----------
    result    : dDTFResult from compute_ddtf_connectivity()
    threshold : minimum normalised strength to display
    suptitle  : optional figure title

    Returns
    -------
    matplotlib Figure  (call .savefig('connectivity.png', dpi=150) to export)
    """
    bands = ["alpha", "beta", "delta", "all"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), facecolor="white")
    axes = axes.flatten()

    for ax, band in zip(axes, bands):
        _plot_ddtf_on_axis(ax, result, band, threshold)

    title = suptitle or "dDTF Band Connectivity"
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def _plot_ddtf_on_axis(ax, result, band, threshold=0.20):
    """Internal helper: render one band panel onto an existing Axes."""
    mat      = result[band]
    ch_names = result["ch_names"]
    n_ch     = len(ch_names)
    max_val  = mat.max()
    norm_mat = mat / max_val if max_val > 0 else mat
    cmap     = plt.get_cmap(BAND_CMAPS[band])

    ax.set_aspect("equal")
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.05, 1.15)
    ax.axis("off")

    head = plt.Circle((0,0), 0.92, fill=False, linewidth=1.2,
                       color="#CCCCCC", zorder=1)
    ax.add_patch(head)
    ax.plot([-.10, 0, .10], [0.92, 1.05, 0.92],
            color="#CCCCCC", linewidth=1.2, zorder=1)
    for side in (-1, 1):
        ax.add_patch(mpatches.Arc((side*0.92, 0), 0.10, 0.22,
             angle=0, theta1=270, theta2=90,
             color="#CCCCCC", linewidth=1.2, zorder=1))

    coords = {ch: SCALP_POS.get(ch, (
        0.7*np.cos(2*np.pi*i/n_ch - np.pi/2),
        0.7*np.sin(2*np.pi*i/n_ch - np.pi/2)
    )) for i, ch in enumerate(ch_names)}

    for i, frm in enumerate(ch_names):
        for j, to in enumerate(ch_names):
            if i == j: continue
            s = norm_mat[j, i]
            if s < threshold: continue
            _draw_arc(ax, coords[frm], coords[to], s, cmap(0.35 + s*0.65))

    out_s = norm_mat.sum(axis=0)
    out_s = out_s / (out_s.max() + 1e-9)
    for ch in ch_names:
        x, y = coords[ch]
        r = 0.045 + out_s[ch_names.index(ch)] * 0.035
        ax.add_patch(plt.Circle((x,y), r, color=cmap(0.6),
                                zorder=5, ec="white", linewidth=1.5))
        ax.text(x, y, ch, ha="center", va="center",
                fontsize=7, fontweight="bold", color="white", zorder=6)

    sm = ScalarMappable(cmap=cmap, norm=Normalize(0, max_val))
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, orientation="horizontal",
                      fraction=0.04, pad=0.01, aspect=25)
    cb.ax.tick_params(labelsize=7)
    ax.set_title(f"{band.capitalize()}  ({BAND_RANGES[band]})",
                 fontsize=11, fontweight="bold", pad=8)

class BandPowerFeatures(TypedDict):
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


def compute_attn_features(raw: mne.io.RawArray) -> BandPowerFeatures:
    """
    Compute frequency-band power features from a 12-channel EEG/EOG recording.

    Expected channel layout
    -----------------------
    EOG : "EOG 1", "EOG 2"
    EEG : "F1", "F2", "C3", "C4", "P3", "P4", "O1", "O2", "Cz", "Pz"

    Features returned
    -----------------
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

    # ------------------------------------------------------------------ #
    # 2.  Per-band power vectors                                          #
    # ------------------------------------------------------------------ #
    alpha_pw = _band_power(psds, freqs, *ALPHA_BAND)   # shape (10,)
    beta_pw  = _band_power(psds, freqs, *BETA_BAND)    # shape (10,)

    return BandPowerFeatures(
        alpha_power          = float(alpha_pw.mean()),
        beta_power           = float(beta_pw.mean()),
    )
