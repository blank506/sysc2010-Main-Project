import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from scipy.signal import butter, filtfilt, firwin, find_peaks
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt

# ---------------------------
# GLOBAL VARIABLES
# ---------------------------
time = None
signal = None
original_signal = None
processed_signal = None
fs = None

# ---------------------------
# PREPROCESSING
# ---------------------------
def preprocess_signal(sig):
    sig = pd.Series(sig).interpolate().bfill().values
    if np.std(sig) == 0:
        return sig
    return (sig - np.mean(sig)) / np.std(sig)

# ---------------------------
# LOAD CSV
# ---------------------------
def load_csv():
    global time, signal, original_signal, processed_signal, fs

    file_path = filedialog.askopenfilename(filetypes=[("CSV Files","*.csv")])
    if not file_path:
        return

    df = pd.read_csv(file_path)

    # -------- TIME --------
    if "time24" in df.columns:
        dt = pd.to_datetime(df["time24"])#for the ECG data/had to change the name from the data file from time to time 24
        time = (dt - dt.iloc[0]).dt.total_seconds().values

    elif "time" in df.columns:
        if np.issubdtype(df["time"].dtype, np.number):
            time = df["time"].values.astype(float)
        else:
            dt = pd.to_datetime(df["time"])
            time = (dt - dt.iloc[0]).dt.total_seconds().values

    elif "t" in df.columns:
        time = df["t"].values.astype(float)

    elif "Datetime" in df.columns:   
         dt = pd.to_datetime(df["Datetime"])#for the TEMP data
         time = (dt - dt.iloc[0]).dt.total_seconds().values

    #elif "Datetime1" in df.columns:  # fallback (not ideal) we already got a column for TEMP time
        #dt = pd.to_datetime(df["Datetime1"], errors='coerce')
        
    elif "PSD_Time_s" in df.columns: 
        time = df["PSD_Time_s"].values.astype(float)

        # Remove duplicates properly (KEEP ALIGNMENT)
        mask = np.diff(time, prepend=time[0]-1) > 0
        time = time[mask]
        #signal = signal[mask]

        # Normalize time
        time = time - time[0]
    else:
        messagebox.showerror("Error", "No valid time column found")
        return


    # checking (optional but useful)
    print("Duration:", time[-1])  
    print("Samples:", len(time))
    print("Duration:", time[-1] - time[0])
    
    # -------- SIGNAL --------
    if signal_type.get() == "Respiration" and "PSD_Flow_L_s" in df.columns:
        sigcol = "PSD_Flow_L_s"

    elif signal_type.get() == "Temperature" and "DAYTON_MW" in df.columns:
        sigcol = "DAYTON_MW"

    elif "ecg" in df.columns:
        sigcol = "ecg"

    elif "signal" in df.columns:
        sigcol = "signal"

    else:
        num_cols = df.select_dtypes(include=[np.number]).columns

        if len(num_cols) == 0:
            messagebox.showerror("Error", "No numeric signal column found")
            return

        sigcol = num_cols[0]

    signal = df[sigcol].values
    signal = pd.Series(signal).interpolate().bfill().ffill().values

    # Apply SAME mask to signal 
    if 'mask' in locals():
        signal = signal[mask]

    # Make signal match cleaned time
    #signal = signal[:len(time)]

    if signal_type.get() == "Temperature":
        sig = pd.Series(signal)
        sig = sig.interpolate().rolling(window=5, center=True).mean()
        original_signal = sig.bfill().ffill().values

    elif signal_type.get() == "Respiration":
        sig = pd.Series(signal)
        sig = sig.rolling(window=15, center=True).mean()
        original_signal = sig.bfill().ffill().values


    else:
        original_signal = preprocess_signal(signal)

    processed_signal = original_signal.copy()

    print("Time length:", len(time))
    print("Signal length:", len(signal))
    # -------- SAMPLING RATE --------
    dt = np.diff(time)
    dt = dt[dt > 0]
    dt = np.diff(time)
    dt = dt[dt > 0]

    if len(dt) == 0:
        fs = 1.0
    else:
        fs = 1 / np.mean(dt)

    if signal_type.get() == "Temperature":
        # Temperature can have large gaps → DO NOT trim
        fs = 1 / np.mean(dt)

    else:
        # ECG & Resp → remove outliers safely
        dt = dt[(dt > np.percentile(dt, 5)) & (dt < np.percentile(dt, 95))]
        fs = 1 / np.mean(dt)

    print("Sampling frequency:", fs)
    plot_time()
# ---------------------------
# FILTERS
# ---------------------------
def butter_filter(data, cutoff, fs, btype):
    nyq = 0.5 * fs
    if isinstance(cutoff, list):
        Wn = [c/nyq for c in cutoff]
    else:
        Wn = cutoff / nyq

    if signal_type.get() == "ECG":
        order = 4 #for ECG
    else:
        order = 6   # stronger for respiration & temp

    b, a = butter(order, Wn, btype=btype)
    return filtfilt(b, a, data)

def fir_filter(data, cutoff, fs, btype):
    nyq = 0.5 * fs

    if btype == "low":
        norm_cutoff = min(cutoff / nyq, 0.99)
        taps = firwin(101, norm_cutoff)

    elif btype == "high":
        norm_cutoff = max(cutoff / nyq, 0.001)
        taps = firwin(101, norm_cutoff, pass_zero=False)

    else:  # band
        low = max(cutoff[0] / nyq, 0.001)
        high = min(cutoff[1] / nyq, 0.99)
        taps = firwin(101, [low, high], pass_zero=False)

    return filtfilt(taps, [1.0], data)


def get_cutoff():
    stype = signal_type.get()
    if stype == "ECG":
        return [0.5, 25] #had to chnage it to 25 insteade of 40, becuase it might help to remove the very hight freq? maybe?
    
    elif stype == "Respiration":
        return [0.1,0.4] #these values are better so that they remove more noise but still keep the signal 
    
    elif stype == "Temperature":
        return [1/(3600*24*10), 1/(3600*6)] #low at 10 days, high at 6 hours 
    
    return [0.5, 25]

def apply_filter(filter_type):
    global processed_signal

    cutoff = get_cutoff()
    method = filter_method.get()

    if filter_type == "low":
        cutoff_val = cutoff[1]
    elif filter_type == "high":
        cutoff_val = cutoff[0]
    
    else:
        cutoff_val = cutoff

    if method == "IIR":
        processed_signal = butter_filter(original_signal, cutoff_val, fs, filter_type)
    else:
        processed_signal = fir_filter(original_signal, cutoff_val, fs, filter_type)

    plot_time()

# ---------------------------
# FEATURES
# ---------------------------
def compute_features():
    stype = signal_type.get()

    # -------- ECG --------
    if stype == "ECG":
        segment = processed_signal[:int(fs*10)]
        peaks, _ = find_peaks(segment, distance=int(fs*0.4))

        if len(peaks) < 2:
            return "Heart Rate: 0 BPM"

        duration = time[peaks[-1]] - time[peaks[0]]
        bpm = len(peaks)/duration * 60

        return f"Heart Rate: {bpm:.2f} BPM"

    # -------- RESPIRATION --------
    elif stype == "Respiration":
        # Find peaks (breaths)
        peaks, _ = find_peaks(processed_signal, distance=int(fs*1.5))

        if len(peaks) < 2:
            return "Breathing Rate: 0 BPM"

        duration = time[peaks[-1]] - time[peaks[0]]
        br = len(peaks)/duration * 60  # breaths per minute

        rms = np.sqrt(np.mean(processed_signal**2))

        return f"Breathing Rate: {br:.2f} BPM\nResp RMS: {rms:.3f}"

    # -------- TEMPERATURE --------
    elif stype == "Temperature":
        mean_temp = np.mean(processed_signal)

        # Trend (slope)
        t = time - time[0]
        slope = np.polyfit(t, processed_signal, 1)[0]
        slope_per_hour = slope * 3600
        return f"Avg Temp: {mean_temp:.2f}\nTrend: {slope_per_hour:.4f} °C/hour"
    return ""

# ---------------------------
# STATS
# ---------------------------
def compute_stats():
    mean = np.mean(processed_signal)
    std = np.std(processed_signal)
    rms = np.sqrt(np.mean(processed_signal**2))
    p2p = np.ptp(processed_signal)

    stats_text.set(
        f"Mean: {mean:.3f}\nSTD: {std:.3f}\nRMS: {rms:.3f}\nP2P: {p2p:.3f}\n{compute_features()}"
    )

# ---------------------------
# FFT
# ---------------------------
def show_fft():
    if original_signal is None or processed_signal is None:
        return

    N = len(original_signal)

    # FFT for raw signal
    yf_raw = fft(original_signal)
    xf = fftfreq(N, 1/fs)

    # FFT for filtered signal
    yf_filtered = fft(processed_signal)

    plt.figure(figsize=(10,5))

    # Raw FFT
    plt.plot(xf[:N//2], np.abs(yf_raw[:N//2]), 
             label="Raw FFT", alpha=0.5)

    # Filtered FFT
    plt.plot(xf[:N//2], np.abs(yf_filtered[:N//2]), 
             label="Filtered FFT", linewidth=2)

    plt.title("FFT Comparison (Raw vs Filtered)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.grid()
    plt.show()

# ---------------------------
# PLOT
# ---------------------------
def plot_time():
    if time is None or original_signal is None:
        return

    stype = signal_type.get()

    # Smart display window
    if stype == "Temperature":
        duration = 3600 * 24 * 30   # show 30 days
    elif stype == "Respiration":
        duration = 30 # show 30 seconds 
    else:
        duration = 10              # show 10 seconds

    idx = time <= duration

    plt.figure()
    plt.plot(time[idx], original_signal[idx], label="Raw", alpha=0.5)

    if processed_signal is not None:
        plt.plot(time[idx], processed_signal[idx], label="Filtered")

    plt.legend()
    plt.title(f"Time Domain ({stype})")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.show()

    compute_stats()
# ---------------------------
# RESET
# ---------------------------
def reset_signal():
    global processed_signal
    processed_signal = original_signal.copy()
    plot_time()

# ---------------------------
# GUI
# ---------------------------
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Signal Processing Tool")
    root.geometry("500x500")

    tk.Label(root, text="Signal Processing Tool", font=("Arial",16)).pack(pady=10)

    tk.Button(root, text="Load CSV", command=load_csv).pack(pady=5)

    signal_type = tk.StringVar(value="ECG")
    ttk.Combobox(root, textvariable=signal_type,
                values=("ECG","Respiration","Temperature")).pack(pady=5)

    filter_method = tk.StringVar(value="IIR")
    ttk.Combobox(root, textvariable=filter_method,
                values=("IIR","FIR")).pack(pady=5)

    frame = tk.Frame(root)
    frame.pack(pady=10)

    tk.Button(frame, text="Low Pass", command=lambda: apply_filter("low")).grid(row=0,column=0,padx=5)
    tk.Button(frame, text="High Pass", command=lambda: apply_filter("high")).grid(row=0,column=1,padx=5)
    tk.Button(frame, text="Band Pass", command=lambda: apply_filter("band")).grid(row=0,column=2,padx=5)

    tk.Button(root, text="Show FFT", command=show_fft).pack(pady=10)
    tk.Button(root, text="Reset", command=reset_signal).pack(pady=5)

    stats_text = tk.StringVar()
    tk.Label(root, textvariable=stats_text).pack(pady=20)

    root.mainloop()