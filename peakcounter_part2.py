import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os

# Path difference
d = 5e-5
threshold = 0.01 # unified threshold (region + peak detection)

# Store results
peak_counts = []   # valid trials
trial_results = [] # all trials (including skipped)

# Ensure output folder exists
out_dir = "results"
os.makedirs(out_dir, exist_ok=True)

# Loop over 10 files
for i in range(1, 11):
    filename = f"part2_data_{i}.csv"
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found, skipping.")
        trial_results.append((i, None))
        continue

    # ---------------- 1) Load CSV (ignore comment lines) ----------------
    all_data = []
    with open(filename, "r") as f:
        for line in f:
            if not line.startswith("%") and line.strip():
                parts = line.strip().split(",")
                try:
                    values = [float(x) for x in parts]
                    all_data.append(values)
                except ValueError:
                    continue

    data = np.array(all_data)
    if data.shape[1] < 2:
        print(f"{filename} does not have at least two columns, skipping.")
        trial_results.append((i, None))
        continue

    # Extract time and probe signal
    t = data[:, 0]
    y = data[:, 1]

    # ---------------- 2) Crop to strong region (y >= threshold) ----------------
    mask = y >= threshold
    if not mask.any():
        print(f"No data points >= {threshold} V in {filename}, skipping.")
        trial_results.append((i, None))
        continue

    first_idx = np.argmax(mask)
    last_idx = len(mask) - 1 - np.argmax(mask[::-1])
    t = t[first_idx:last_idx+1]
    y = y[first_idx:last_idx+1]

    # ---------------- 3) Detect peaks above threshold ----------------
    peaks, props = find_peaks(y, height=threshold)

    # ---------------- 4) Remove peaks too close together ----------------
    total_time = t[-1] - t[0]
    min_sep = 0.003 * total_time   # 0.3% of duration

    filtered_peaks = []
    last_time = -np.inf
    for p in peaks:
        tp = t[p]
        if tp - last_time > min_sep:
            filtered_peaks.append(p)
            last_time = tp

    m = len(filtered_peaks)
    if m > 0:
        wavelength = 2 * d / m
        peak_counts.append(m)
        trial_results.append((i, m))
    else:
        trial_results.append((i, 0))

        # ---------------- 5) Plot and save ----------------
    plt.figure(figsize=(10, 5))
    plt.plot(t, y, label="Probe Signal", color="blue")

    # Draw horizontal threshold line
    plt.axhline(threshold, color="gray", linestyle="--", linewidth=1, 
                label=f"Threshold ({threshold} V)")

    # Plot all detected peaks (before filtering)
    plt.plot(t[peaks], y[peaks], "x", color="orange", label="All Peaks (raw)")

    # Plot filtered peaks (after min separation)
    plt.plot(t[filtered_peaks], y[filtered_peaks], "ro", label="Accepted Peaks")

    plt.xlabel("Time [s]", fontsize=16)
    plt.ylabel("Probe Voltage [V]", fontsize=16)

    # Use descriptive title with trial number
    if m > 0:
        plt.title(
            f"Recorded voltage from detector vs time (Trial {i})\n"
            f"m = {m}, λ ≈ {2*d/m:.2e} m",
            fontsize=16
        )
    else:
        plt.title(
            f"Recorded voltage from detector vs time (Trial {i})\n"
            f"No valid peaks",
            fontsize=16
        )

    # Place legend outside on the right
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()

    plt.savefig(os.path.join(out_dir, f"{filename}_peaks.png"), bbox_inches="tight")
    plt.close()



# ---------------- 6) Summary statistics ----------------
print("\nFringe counts per trial:")
for trial, m in trial_results:
    if m is None:
        print(f" Trial {trial}: skipped (no valid data)")
    elif m == 0:
        print(f" Trial {trial}: no valid peaks")
    else:
        print(f" Trial {trial}: m = {m}")

if peak_counts:
    peak_counts = np.array(peak_counts)

    # Average and standard error of fringe counts
    avg_peaks = np.mean(peak_counts)
    std_peaks = np.std(peak_counts, ddof=1)   # sample standard deviation
    stderr_peaks = std_peaks / np.sqrt(len(peak_counts))  # standard error of mean

    # Wavelength from mean fringe count
    wavelength_mean = 2 * d / avg_peaks

    # Error propagation: λ = 2d/m
    d_error = 1e-7
    m_avg = avg_peaks
    sigma_m = stderr_peaks
    stderr_wavelength = np.sqrt(
        (2 / m_avg * d_error)**2 +   # contribution from d
        (2 * d / m_avg**2 * sigma_m)**2   # contribution from m
    )

    print("\nSummary across all valid trials:")
    print(f"Average fringe count (m): {avg_peaks:.2f}")
    print(f"Standard error of fringe count: {stderr_peaks:.2f}")
    print(f"Experimental wavelength: {wavelength_mean:.3e} m")
    print(f"Propagated error in wavelength: {stderr_wavelength:.3e} m")

else:
    print("No valid data found across files.")