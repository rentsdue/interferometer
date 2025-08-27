import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt


# ---------------- Hysteresis crossing function ----------------
def find_negative_to_positive_transitions(time_series, values, lower_threshold, upper_threshold):
    """Detect rising threshold crossings with hysteresis."""

    # Ensure inputs are numpy arrays for efficient indexing and vectorized operations
    t = np.asarray(time_series)
    x = np.asarray(values)

    # Check that time and signal arrays are the same length
    if len(t) != len(x):
        raise ValueError("time_series and values must have the same length.")
    # Check that thresholds are valid (lower must be strictly less than upper)
    if lower_threshold >= upper_threshold:
        raise ValueError("lower_threshold must be strictly less than upper_threshold.")

    # ---------------- Initialize state machine ----------------
    # Determine initial signal state based on the first sample
    v0 = x[0]
    if v0 <= lower_threshold:
        state = 'below'   # signal starts below lower threshold
    elif v0 >= upper_threshold:
        state = 'above'   # signal starts above upper threshold
    else:
        state = 'unknown' # signal starts between thresholds, state undecided

    # Lists to store times and values where crossings occur
    times, vals = [], []

    # ---------------- Process the signal sample by sample ----------------
    for ti, vi in zip(t[1:], x[1:]):  # iterate over subsequent samples
        if state == 'unknown':
            # If initial state was unclear, wait until signal leaves dead-band
            if vi <= lower_threshold:
                state = 'below'
            elif vi >= upper_threshold:
                state = 'above'

        elif state == 'below':
            # If currently below the lower threshold, watch for upward crossing
            if vi >= upper_threshold:
                # Crossing detected: record time and value
                times.append(ti)
                vals.append(vi)
                # Update state to "above" once crossing occurs
                state = 'above'

        elif state == 'above':
            # If currently above the upper threshold, watch for downward crossing
            if vi <= lower_threshold:
                # No event recorded here, just reset state to "below"
                state = 'below'

    # Convert results to numpy arrays for convenient use downstream
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

    # ---------------- Hysteresis method ----------------
    tt, xx = find_negative_to_positive_transitions(t, x_filt, lower_threshold, upper_threshold)
    N_hysteresis = len(tt)

    # ---------------- Wavelength calculation ----------------
    displacement_m = 50e-6  # Mirror moved 50 micrometers
    wavelength_hyst = 2 * displacement_m / N_hysteresis if N_hysteresis > 0 else np.nan

    # ---------------- Plot ----------------
    plt.figure(figsize=(12, 6))
    plt.plot(t, x, label="Raw Signal", alpha=0.5)
    plt.plot(t, x_filt, label="Filtered Signal", lw=1.2)

    plt.axhline(lower_threshold, color="gray", ls="--", lw=0.8)
    plt.axhline(upper_threshold, color="gray", ls="--", lw=0.8)

    plt.scatter(tt, xx, color="red", label="Hysteresis crossings", zorder=3)

    plt.xlabel("Time [s]")
    plt.ylabel("Voltage [V]")
    plt.title("Fringe Detection (Hysteresis Method)")
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
