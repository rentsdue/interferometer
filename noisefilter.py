import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os


# ---------------- 1) Load CSV (ignore comment lines) ----------------
all_data = []
with open("0to2_2.csv", "r") as f:
    for line in f:
        if not line.startswith("%") and line.strip():
            parts = line.strip().split(",")
            try:
                values = [float(x) for x in parts]
                all_data.append(values)
            except ValueError:
                # skip lines with non-numeric values
                continue

data = np.array(all_data)

if data.shape[1] < 2:
    raise ValueError("CSV does not have at least two columns (time, probe).")

# Extract time and probe signal
t = data[:, 0]
y = data[:, 1]

# ---------------- 2) Keep only strong region (y >= 0.25) ----------------
mask = y >= 0.000001
if not mask.any():
    print("No data points found with probe_v >= 0.25")
    exit()

first_idx = np.argmax(mask)                  # first True
last_idx = len(mask) - 1 - np.argmax(mask[::-1])  # last True
t = t[first_idx:last_idx+1]
y = y[first_idx:last_idx+1]

# ---------------- 3) Detect peaks above threshold ----------------
peaks, props = find_peaks(y, height=0.25)

# ---------------- 4) Remove peaks too close together (within 0.25 s) ----------------
filtered_peaks = []
last_time = -np.inf
for p in peaks:
    tp = t[p]
    if tp - last_time > 0:
        filtered_peaks.append(p)
        last_time = tp

# ---------------- 5) Plot results ----------------
plt.figure(figsize=(10, 5))
plt.plot(t, y, label="Probe Signal")
plt.plot(t[filtered_peaks], y[filtered_peaks], "ro", label="Detected Peaks")

plt.xlabel("Time [s]")
plt.ylabel("Probe Voltage [V]")
plt.title("Peak Detection (Strong Region Only)")
plt.legend()
plt.tight_layout()
plt.show()

# Results
print(f"Number of detected peaks: {len(filtered_peaks)}")