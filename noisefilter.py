import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os

# Path difference
d = 5e-6

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

    # ---------------- 2) Keep only strong region (y >= 0.25) ----------------
    mask = y >= 0.5
    if not mask.any():
        print(f"No data points >= 0.25 in {filename}, skipping.")
        continue

    first_idx = np.argmax(mask)
    last_idx = len(mask) - 1 - np.argmax(mask[::-1])
    t = t[first_idx:last_idx+1]
    y = y[first_idx:last_idx+1]

    # ---------------- 3) Detect peaks above threshold ----------------
    peaks, props = find_peaks(y, height=0.25)

    # ---------------- 4) Remove peaks too close together (within 0.25 s) ----------------
    total_time = t[-1] - t[0]
    min_sep = 0.04 * total_time   # e.g. require 4% of total duration between peaks

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
    plt.plot(t, y, label="Probe Signal")
    plt.plot(t[filtered_peaks], y[filtered_peaks], "ro", label="Detected Peaks")
    plt.xlabel("Time [s]")
    plt.ylabel("Probe Voltage [V]")
    plt.title(f"Peak Detection - {filename}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{filename}_peaks.png"))
    plt.close()

    # Print per file result
    print(f"{filename}: peaks = {m}, wavelength = {wavelength:.3e} m")

# ---------------- 6) Summary statistics ----------------
if peak_counts:
    avg_peaks = np.mean(peak_counts)
    std_peaks = np.std(peak_counts, ddof=1)  # sample std
    stderr_peaks = std_peaks / np.sqrt(len(peak_counts))

    avg_wavelength = np.mean(wavelengths)
    std_wavelength = np.std(wavelengths, ddof=1)
    stderr_wavelength = std_wavelength / np.sqrt(len(wavelengths))

    print("\nSummary across all files:")
    print(f"Peaks: mean = {avg_peaks:.2f}, std = {std_peaks:.2f}, stderr = {stderr_peaks:.2f}")
    print(f"Wavelength: mean = {avg_wavelength:.3e} m, std = {std_wavelength:.3e}, stderr = {stderr_wavelength:.3e}")
else:
    print("No valid data found across files.")
