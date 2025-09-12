import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import brentq
import math
import os
import csv

# ---------------- Configuration (EDIT THESE TWO) ----------------
lambda_m = 521e-9         # Laser wavelength in meters (EDIT: use your measured or nominal λ)
d_slab_m = 8.0e-3         # Slab thickness in meters (EDIT: set to your slab thickness)

# Peak detection parameters
strong_threshold = 0.25   # V; keep only region where y >= this value
peak_height = 0.25        # V; minimum peak height

# Output
out_dir = "results_n_estimation"
os.makedirs(out_dir, exist_ok=True)
summary_csv = os.path.join(out_dir, "refractive_index_summary.csv")

# ---------------- File groups and angle sweeps ----------------
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
    return arr[:,0], arr[:,1]

# ---------------- Utility: trim to strong region ----------------
def trim_strong_region(t, y, threshold):
    mask = y >= threshold
    if not mask.any():
        return None, None
    first_idx = np.argmax(mask)
    last_idx = len(mask) - 1 - np.argmax(mask[::-1])
    return t[first_idx:last_idx+1], y[first_idx:last_idx+1]

# ---------------- Utility: peak detection ----------------
def detect_filtered_peaks(t, y, height=0.25, min_sep_fraction=0.04):
    peaks, _ = find_peaks(y, height=height)
    if len(peaks) == 0:
        return []
    total_time = t[-1] - t[0]
    min_sep = min_sep_fraction * total_time
    filtered = []
    last_time = -np.inf
    for p in peaks:
        tp = t[p]
        if tp - last_time > min_sep:
            filtered.append(p)
            last_time = tp
    return filtered

# ---------------- Physics model ----------------
def F(n, theta_rad):
    s = math.sin(theta_rad)
    c = math.cos(theta_rad)
    inside = n*n - s*s
    if inside <= 0:
        return float("nan")
    return math.sqrt(inside) - c

def m_predicted(n, theta_a_deg, theta_b_deg, d_slab, wavelength):
    ta = math.radians(theta_a_deg)
    tb = math.radians(theta_b_deg)
    return 2.0 * (d_slab / wavelength) * (F(n, tb) - F(n, ta))

def invert_n_from_m(m_meas, theta_a_deg, theta_b_deg, d_slab, wavelength,
                    n_lo=1.0001, n_hi=2.5):
    def g(n):
        return m_predicted(n, theta_a_deg, theta_b_deg, d_slab, wavelength) - m_meas
    f_lo = g(n_lo)
    f_hi = g(n_hi)
    tries = 0
    while np.sign(f_lo) == np.sign(f_hi) and n_hi < 5.0 and tries < 20:
        n_hi += 0.25
        f_hi = g(n_hi)
        tries += 1
    if np.sign(f_lo) == np.sign(f_hi):
        raise RuntimeError("Could not bracket root for n.")
    return brentq(g, n_lo, n_hi, xtol=1e-9, rtol=1e-9, maxiter=200)

# ---------------- Main loop ----------------
rows = []
grouped_results = {2: [], 4: [], 6: []}
grouped_m = {2: [], 4: [], 6: []}

# Per-group trial counters (reset per angle)
trial_counters = {2: 1, 4: 1, 6: 1}

print("\n===== Trial Results =====")
for fname, theta_a_deg, theta_b_deg in file_specs:
    if not os.path.exists(fname):
        print(f"{fname}: file not found; skipping.")
        continue

    try:
        t, y = load_two_columns_csv(fname)
    except Exception as e:
        print(f"{fname}: error reading file ({e}).")
        continue

    t_trim, y_trim = trim_strong_region(t, y, strong_threshold)
    if t_trim is None:
        print(f"{fname}: no region ≥ {strong_threshold:.2f} V; skipping.")
        continue

    idx = detect_filtered_peaks(t_trim, y_trim, peak_height, min_sep_fraction=0.04)
    m_meas = int(len(idx))

    try:
        n_est = invert_n_from_m(m_meas, theta_a_deg, theta_b_deg, d_slab_m, lambda_m)
    except Exception as e:
        n_est = np.nan

    # Group-specific trial number
    angle_group = int(theta_b_deg)
    trial_number = trial_counters[angle_group]

    # ---------------- Plot and save ----------------
    plt.figure(figsize=(10, 5))
    plt.plot(t_trim, y_trim, label="Probe Signal", color="blue")

    plt.axhline(strong_threshold, color="gray", linestyle="--", linewidth=1,
                label=f"Threshold ({strong_threshold} V)")

    all_peaks, _ = find_peaks(y_trim, height=peak_height)
    plt.plot(t_trim[all_peaks], y_trim[all_peaks], "x", color="orange", label="All Peaks (raw)")

    if len(idx) > 0:
        plt.plot(t_trim[idx], y_trim[idx], "ro", label="Accepted Peaks")

    plt.xlabel("Time [s]", fontsize=16)
    plt.ylabel("Voltage [V]", fontsize=16)

    if m_meas > 0:
        plt.title(f"Recorded voltage from detector vs time (0→{angle_group}° Trial {trial_number})\n"
                  f"(m={m_meas}, λ≈{2*d_slab_m/m_meas:.2e} m)", fontsize=16)
    else:
        plt.title(f"0→{angle_group}° Trial {trial_number}: Voltage vs Time graph - peak detection\n(no valid peaks)", fontsize=16)

    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()

    plot_path = os.path.join(out_dir, f"{os.path.splitext(fname)[0]}_peaks.png")
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()

    # Record
    rows.append({
        "file": fname,
        "theta_start_deg": theta_a_deg,
        "theta_end_deg": theta_b_deg,
        "trial_number": trial_number,
        "peaks_m": m_meas,
        "n_estimate": n_est,
        "plot": plot_path
    })

    grouped_m[int(theta_b_deg)].append(m_meas)
    if np.isfinite(n_est):
        grouped_results[int(theta_b_deg)].append(n_est)

    if np.isfinite(n_est):
        print(f"{fname}: Trial {trial_number} (0→{angle_group}°) | m = {m_meas}, n = {n_est:.5f}")
    else:
        print(f"{fname}: Trial {trial_number} (0→{angle_group}°) | m = {m_meas}, n = NaN")

    # Increment this group’s trial counter
    trial_counters[angle_group] += 1

# ---------------- Summary ----------------
def summarize(values):
    arr = np.array([x for x in values if np.isfinite(x)], dtype=float)
    if arr.size == 0:
        return np.nan, np.nan
    mean = np.mean(arr)
    sem = np.std(arr, ddof=1)/math.sqrt(arr.size) if arr.size > 1 else 0.0
    return mean, sem

print("\n===== Summary by Angle Range =====")
for angle in (2, 4, 6):
    mean_m, sem_m = summarize(grouped_m[angle])
    mean_n, sem_n = summarize(grouped_results[angle])
    print(f"0→{angle}°:")
    print(f"  Average m = {mean_m:.2f} ± {sem_m:.2f}")
    print(f"  Average n = {mean_n:.5f} ± {sem_n:.5f}")

# ---------------- Save CSV ----------------
with open(summary_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["file", "theta_start_deg", "theta_end_deg", "trial_number", "peaks_m", "n_estimate", "plot_path"])
    for r in rows:
        writer.writerow([r["file"], r["theta_start_deg"], r["theta_end_deg"], r["trial_number"], r["peaks_m"], r["n_estimate"], r["plot"]])

print(f"\nSummary written to: {summary_csv}")
print(f"Plots saved under: {out_dir}/")
