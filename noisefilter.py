import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt


# Load while skipping comments
data = np.loadtxt("test1.csv", comments="%", delimiter=",")

t = data[:, 0]
x = data[:, 1]

def find_negative_to_positive_transitions(time_series, values, lower_threshold, upper_threshold):
    """Detect rising threshold crossings with hysteresis."""
    t = np.asarray(time_series)
    x = np.asarray(values)

    if len(t) != len(x):
        raise ValueError("time_series and values must have the same length.")
    if lower_threshold >= upper_threshold:
        raise ValueError("lower_threshold must be strictly less than upper_threshold.")

    # Initialize state
    v0 = x[0]
    if v0 <= lower_threshold:
        state = 'below'
    elif v0 >= upper_threshold:
        state = 'above'
    else:
        state = 'unknown'

    times, vals = [], []

    for ti, vi in zip(t[1:], x[1:]):
        if state == 'unknown':
            if vi <= lower_threshold:
                state = 'below'
            elif vi >= upper_threshold:
                state = 'above'
        elif state == 'below':
            if vi >= upper_threshold:
                times.append(ti)
                vals.append(vi)
                state = 'above'
        elif state == 'above':
            if vi <= lower_threshold:
                state = 'below'
    return np.array(times), np.array(vals)


# ---------------- Main script ----------------
if __name__ == "__main__":
    # ---------------- Load CSV ----------------
    all_data = []
    with open("test1.csv", "r") as f:
        for line in f:
            if not line.startswith("%") and line.strip():
                all_data.append([float(x) for x in line.strip().split(",")])

    data = np.array(all_data)

    # Extract columns: Time and Probe A
    t = data[:, 0]
    x = data[:, 1]

    # Optional smoothing (comment out if unnecessary)
    b, a = butter(3, 0.05)  # low-pass filter
    x_filt = filtfilt(b, a, x)

    # ---------------- Automatic thresholds (midpoint ± hysteresis) ----------------
    signal_min, signal_max = np.min(x_filt), np.max(x_filt)
    mid = 0.5 * (signal_min + signal_max)
    hyst = 0.002  # 2 mV
    lower_threshold = mid - hyst
    upper_threshold = mid + hyst

    print(f"Signal range: {signal_min:.3f} to {signal_max:.3f} V")
    print(f"Midpoint: {mid:.3f} V, thresholds = [{lower_threshold:.3f}, {upper_threshold:.3f}]")

    # ---------------- Method 1: Zero-crossing with hysteresis ----------------
    tt, xx = find_negative_to_positive_transitions(t, x_filt, lower_threshold, upper_threshold)
    N_hysteresis = len(tt)

    # ---------------- Method 2: Peak detection ----------------
    peaks, _ = find_peaks(x_filt, prominence=0.002)  # adjust as needed
    troughs, _ = find_peaks(-x_filt, prominence=0.002)
    N_peaks = len(peaks) + len(troughs)  # count both maxima & minima

    # ---------------- Wavelength calculation ----------------
    displacement_m = 50e-6  # Mirror moved 50 micrometers
    wavelength_hyst = 2 * displacement_m / N_hysteresis if N_hysteresis > 0 else np.nan
    wavelength_peaks = 2 * displacement_m / N_peaks if N_peaks > 0 else np.nan

    # ---------------- Plot ----------------
    plt.figure(figsize=(12, 6))
    plt.plot(t, x, label="Raw Signal", alpha=0.5)
    plt.plot(t, x_filt, label="Filtered Signal", lw=1.2)

    plt.axhline(lower_threshold, color="gray", ls="--", lw=0.8)
    plt.axhline(upper_threshold, color="gray", ls="--", lw=0.8)

    plt.scatter(tt, xx, color="red", label="Hysteresis crossings", zorder=3)
    plt.scatter(t[peaks], x_filt[peaks], color="blue", marker="x", label="Peaks")
    plt.scatter(t[troughs], x_filt[troughs], color="green", marker="x", label="Troughs")

    plt.xlabel("Time [s]")
    plt.ylabel("Voltage [V]")
    plt.title("Fringe Detection (Two Methods)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ---------------- Print results ----------------
    print("=== Signal Diagnostics ===")
    print(f"Samples: {len(x)}")
    print(f"Duration: {t[-1] - t[0]:.6f} s")
    print(f"Mean: {np.mean(x):.6f} V")
    print(f"Std deviation: {np.std(x):.6f} V")
    print(f"Min: {np.min(x):.6f} V")
    print(f"Max: {np.max(x):.6f} V")
    print(f"Peak-to-peak: {np.ptp(x):.6f} V")

    # FFT check
    X = np.fft.rfft(x - np.mean(x))
    freqs = np.fft.rfftfreq(len(x), d=(t[1]-t[0]))
    dominant_freq = freqs[np.argmax(np.abs(X))]
    print(f"Dominant frequency component: {dominant_freq:.2f} Hz")


    print("\n=== Fringe Count Results ===")
    print(f"Hysteresis method: {N_hysteresis} fringes → λ ≈ {wavelength_hyst*1e9:.2f} nm")
    print(f"Peak method:       {N_peaks} extrema  → λ ≈ {wavelength_peaks*1e9:.2f} nm")
