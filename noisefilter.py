import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os

# Path difference
d = 5e-6
threshold = 0.25  # unified threshold (region + peak detection)

# Store results
peak_counts = []
wavelengths = []

# Ensure output folder exists
out_dir = "results"
os.makedirs(out_dir, exist_ok=True)

# Loop over 10 files
for i in range(1, 11):
    filename = f"c1_data_{i}.csv"
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found, skipping.")
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
        continue

    # Extract time and probe signal
    t = data[:, 0]
    y = data[:, 1]

    # ---------------- 2) Crop to strong region (y >= threshold) ----------------
    mask = y >= threshold
    if not mask.any():
        print(f"No data points >= {threshold} V in {filename}, skipping.")
        continue

    first_idx = np.argmax(mask)
    last_idx = len(mask) - 1 - np.argmax(mask[::-1])
    t = t[first_idx:last_idx+1]
    y = y[first_idx:last_idx+1]

    # ---------------- 3) Detect peaks above threshold ----------------
    peaks, props = find_peaks(y, height=threshold)

    # ---------------- 4) Remove peaks too close together ----------------
    total_time = t[-1] - t[0]
    min_sep = 0.04 * total_time   # 4% of duration

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
        wavelengths.append(wavelength)

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

    plt.xlabel("Time [s]")
    plt.ylabel("Probe Voltage [V]")
    plt.title(f"Peak Detection - {filename}\n(m={m}, λ={wavelength:.2e} m)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{filename}_peaks.png"))
    plt.close()

# ---------------- 6) Summary statistics ----------------
if peak_counts:
    avg_peaks = np.mean(peak_counts)
    std_peaks = np.std(peak_counts, ddof=1)  # sample std
    stderr_peaks = std_peaks / np.sqrt(len(peak_counts))

    wavelengths = np.array(wavelengths)
    avg_wavelength = np.mean(wavelengths)

    # Error propagation
    d_error = 1e-7  # error in d
    sigma_m = stderr_peaks
    m_avg = avg_peaks
    stderr_wavelength = np.sqrt((2 / m_avg * d_error)**2 +
                                (2 * d / m_avg**2 * sigma_m)**2)

    print("\nSummary across all files:")
    print(f"Peaks: mean = {avg_peaks:.2f}, std = {std_peaks:.2f}, stderr = {stderr_peaks:.2f}")
    print(f"Wavelength: mean = {avg_wavelength:.3e} m, stderr (propagated) = {stderr_wavelength:.3e} m")
else:
    print("No valid data found across files.")
