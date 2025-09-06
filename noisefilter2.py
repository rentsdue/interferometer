import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import brentq
import math
import os
import csv

# ---------------- Configuration (EDIT THESE TWO) ----------------
lambda_m = 532e-9         # Laser wavelength in meters (EDIT: use your measured or nominal λ)
d_slab_m = 8.0e-3         # Slab thickness in meters (EDIT: set to your slab thickness)

# Peak detection parameters
strong_threshold = 0.25   # V; keep only region where y >= this value
peak_height = 0.25         # V; minimum peak height
min_peak_separation_s = 0.25  # s; reject peaks closer than this in time

# Output
out_dir = "results_n_estimation"
os.makedirs(out_dir, exist_ok=True)
summary_csv = os.path.join(out_dir, "refractive_index_summary.csv")

# ---------------- File groups and angle sweeps ----------------
# We interpret the filename prefixes as angular sweeps: 0→2°, 0→4°, 0→6°.
# Angles are in degrees here and converted to radians for computation.
file_specs = []
file_specs += [(f"0to2_{i}.csv", 0.0, 2.0) for i in (2, 4, 6)]
file_specs += [(f"0to4_{i}.csv", 0.0, 4.0) for i in (2, 4, 6)]
file_specs += [(f"0to6_{i}.csv", 0.0, 6.0) for i in (1, 3, 5)]

# ---------------- Utility: load CSV ignoring '%' comments ----------------
def load_two_columns_csv(path):
    all_rows = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("%"):
                continue
            parts = s.split(",")
            try:
                vals = [float(x) for x in parts]
            except ValueError:
                continue
            if len(vals) >= 2:
                all_rows.append(vals[:2])
    if not all_rows:
        raise ValueError(f"No valid numeric data in {path}")
    arr = np.array(all_rows, dtype=float)
    if arr.shape[1] < 2:
        raise ValueError(f"{path} does not contain at least two numeric columns")
    return arr[:,0], arr[:,1]

# ---------------- Utility: trim to strong region ----------------
def trim_strong_region(t, y, threshold):
    mask = y >= threshold
    if not mask.any():
        return None, None
    first_idx = np.argmax(mask)
    last_idx = len(mask) - 1 - np.argmax(mask[::-1])
    return t[first_idx:last_idx+1], y[first_idx:last_idx+1]

# ---------------- Utility: peak detection with time-spacing filter ----------------
def detect_filtered_peaks(t, y, height=0.25, min_sep_fraction=0.04):
    """
    Detect peaks with minimum separation defined as a fraction of total duration.
    """
    peaks, _ = find_peaks(y, height=height)
    if len(peaks) == 0:
        return []

    total_time = t[-1] - t[0]
    min_sep = min_sep_fraction * total_time

    filtered_peaks = []
    last_time = -np.inf
    for p in peaks:
        tp = t[p]
        if tp - last_time > min_sep:
            filtered_peaks.append(p)
            last_time = tp
    return filtered_peaks



# ---------------- Physics model functions ----------------
def F(n, theta_rad):
    # F(n, theta) = sqrt(n^2 - sin^2 theta) - cos theta
    s = math.sin(theta_rad)
    c = math.cos(theta_rad)
    inside = n*n - s*s
    if inside <= 0:
        # Outside physical domain (total internal), but our theta_a → theta_b are small; keep guard.
        return float("nan")
    return math.sqrt(inside) - c

def m_predicted(n, theta_a_deg, theta_b_deg, d_slab, wavelength):
    ta = math.radians(theta_a_deg)
    tb = math.radians(theta_b_deg)
    return 2.0 * (d_slab / wavelength) * (F(n, tb) - F(n, ta))

def invert_n_from_m(m_meas, theta_a_deg, theta_b_deg, d_slab, wavelength,
                    n_lo=1.0001, n_hi=2.5):
    """
    Solve m_predicted(n) = m_meas for n in [n_lo, n_hi] using brentq.
    Adjust n_hi if needed to ensure a sign change.
    """
    def g(n):
        return m_predicted(n, theta_a_deg, theta_b_deg, d_slab, wavelength) - m_meas

    # Ensure sign change by expanding n_hi if necessary (up to a safe ceiling).
    f_lo = g(n_lo)
    f_hi = g(n_hi)
    tries = 0
    while np.sign(f_lo) == np.sign(f_hi) and n_hi < 5.0 and tries < 20:
        n_hi += 0.25
        f_hi = g(n_hi)
        tries += 1

    if np.sign(f_lo) == np.sign(f_hi):
        raise RuntimeError("Could not bracket a root for n; check inputs (m, angles, d, lambda).")

    return brentq(g, n_lo, n_hi, xtol=1e-9, rtol=1e-9, maxiter=200)

# ---------------- Main loop ----------------
rows = []
grouped_results = {2: [], 4: [], 6: []}   # store n_estimates by angle range
grouped_m = {2: [], 4: [], 6: []}         # store m counts

for fname, theta_a_deg, theta_b_deg in file_specs:
    if not os.path.exists(fname):
        print(f"Warning: {fname} not found; skipping.")
        continue

    try:
        t, y = load_two_columns_csv(fname)
    except Exception as e:
        print(f"Error reading {fname}: {e}")
        continue

    t_trim, y_trim = trim_strong_region(t, y, strong_threshold)
    if t_trim is None:
        print(f"{fname}: no data ≥ {strong_threshold:.2f} V; skipping.")
        continue

    idx = detect_filtered_peaks(t_trim, y_trim, peak_height, min_sep_fraction=0.04)
    m_meas = int(len(idx))

    # Numerical inversion for n
    try:
        n_est = invert_n_from_m(
            m_meas=m_meas,
            theta_a_deg=theta_a_deg,
            theta_b_deg=theta_b_deg,
            d_slab=d_slab_m,
            wavelength=lambda_m,
            n_lo=1.0001,
            n_hi=2.0
        )
    except Exception as e:
        n_est = np.nan
        print(f"{fname}: could not estimate n ({e}).")

    # Save plot
    plt.figure(figsize=(10, 5))
    plt.plot(t_trim, y_trim, label="Probe Signal")
    if m_meas > 0:
        plt.plot(t_trim[idx], y_trim[idx], "ro", label="Detected Peaks")
    plt.xlabel("Time [s]")
    plt.ylabel("Probe Voltage [V]")
    plt.title(f"Peak Detection — {fname} (θ: {theta_a_deg}°→{theta_b_deg}°)")
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(out_dir, f"{os.path.splitext(fname)[0]}_peaks.png")
    plt.savefig(plot_path)
    plt.close()

    # Record
    rows.append({
        "file": fname,
        "theta_start_deg": theta_a_deg,
        "theta_end_deg": theta_b_deg,
        "peaks_m": m_meas,
        "n_estimate": n_est,
        "plot": plot_path
    })

    # Group results by final angle (2, 4, or 6 degrees)
    end_angle = int(theta_b_deg)
    if np.isfinite(n_est):
        grouped_results[end_angle].append(n_est)
    grouped_m[end_angle].append(m_meas)

    if np.isfinite(n_est):
        print(f"{fname}: m = {m_meas:3d}, θ = {theta_a_deg:.1f}°→{theta_b_deg:.1f}°, n = {n_est:.5f}")
    else:
        print(f"{fname}: m = {m_meas:3d}, θ = {theta_a_deg:.1f}°→{theta_b_deg:.1f}°, n = NaN")

# ---------------- Summary statistics ----------------
def summarize(values):
    arr = np.array([x for x in values if np.isfinite(x)], dtype=float)
    if arr.size == 0:
        return np.nan, np.nan, np.nan
    mean = np.mean(arr)
    std = np.std(arr, ddof=1) if arr.size > 1 else 0.0
    sem = std / math.sqrt(arr.size) if arr.size > 1 else 0.0
    return mean, std, sem

print("\n===== Summary by Angle Range =====")
for angle in (2, 4, 6):
    mean_m, std_m, sem_m = summarize(grouped_m[angle])
    mean_n, std_n, sem_n = summarize(grouped_results[angle])
    print(f"0→{angle}°:")
    print(f"  Peaks m: mean = {mean_m:.2f}, std = {std_m:.2f}, stderr = {sem_m:.2f}")
    print(f"  Refractive index n: mean = {mean_n:.5f}, std = {std_n:.5f}, stderr = {sem_n:.5f}")

# ---------------- Write CSV summary ----------------
with open(summary_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["file", "theta_start_deg", "theta_end_deg", "peaks_m", "n_estimate", "plot_path"])
    for r in rows:
        writer.writerow([r["file"], r["theta_start_deg"], r["theta_end_deg"], r["peaks_m"], r["n_estimate"], r["plot"]])

print(f"\nSummary written to: {summary_csv}")
print(f"Plots saved under: {out_dir}/")

