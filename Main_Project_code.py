import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, ttk
from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt

# ---------------------------
# GLOBAL VARIABLES
# ---------------------------

time = None
signal = None
processed_signal = np.array([])
fs = 0.0


# ---------------------------
# CSV LOADER
# ---------------------------

def load_csv():

    global time, signal, processed_signal, fs

    file_path = filedialog.askopenfilename(filetypes=[("CSV Files","*.csv")])

    if not file_path:
        return

    df = pd.read_csv(file_path)

    columns = list(df.columns)

    # detect time column
    if "time" in columns:
        time = df["time"].values
    elif "t" in columns:
        time = df["t"].values
    else:
        time = df.iloc[:,0].values

    # choose signal column
    if "signal" in columns:
        signal = df["signal"].values
    elif "lead_I" in columns:
        signal = df["lead_I"].values
    else:
        signal = df.iloc[:,1].values

    # handle missing values
    signal = pd.Series(signal).interpolate().bfill().values

    processed_signal = signal.copy()

    fs = 1/(time[1]-time[0])

    plot_time()
# ---------------------------
# FILTERS
# ---------------------------

def butter_filter(data, cutoff, fs: float, btype, order=4):

    data = np.asarray(data, dtype=float)

    nyq = 0.5 * fs
    normal_cutoff = np.array(cutoff, dtype=float) / nyq

    b, a = butter(order, normal_cutoff, btype)

    return filtfilt(b, a, data)

def apply_lpf():

    global processed_signal
    processed_signal = butter_filter(signal, 40, fs, 'low')
    plot_time()


def apply_hpf():

    global processed_signal
    processed_signal = butter_filter(signal, 0.5, fs, 'high')
    plot_time()


def apply_bpf():

    global processed_signal
    processed_signal = butter_filter(signal, [0.5, 40], fs, 'band')
    plot_time()


# ---------------------------
# STATISTICS
# ---------------------------

def compute_stats():

    mean = np.mean(processed_signal)
    std = np.std(processed_signal)
    rms = np.sqrt(np.mean(processed_signal**2))
    p2p = np.ptp(processed_signal)

    stats_text.set(
        f"Mean: {mean:.3f}\n"
        f"STD: {std:.3f}\n"
        f"RMS: {rms:.3f}\n"
        f"Peak-Peak: {p2p:.3f}"
    )


# ---------------------------
# FFT
# ---------------------------

def show_fft():

    N = len(processed_signal)

    yf = fft(processed_signal)
    xf = fftfreq(N, 1/fs)

    plt.figure()

    plt.plot(xf[:N//2], np.abs(yf[:N//2]))
    plt.title("Frequency Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")

    plt.show()


# ---------------------------
# PLOT TIME SIGNAL
# ---------------------------

def plot_time():

    plt.figure()

    plt.plot(time, signal, label="Raw Signal", alpha=0.5)
    plt.plot(time, processed_signal, label="Processed Signal")

    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.title("Time Domain Signal")
    plt.legend()

    plt.show()

    compute_stats()


# ---------------------------
# RESET
# ---------------------------

def reset_signal():

    global processed_signal
    processed_signal = signal.copy()
    plot_time()


# ---------------------------
# GUI
# ---------------------------

root = tk.Tk()
root.title("Sensor Signal Analysis System")
root.geometry("500x450")

title = tk.Label(root, text="Sensor Signal Processing Tool", font=("Arial",16))
title.pack(pady=10)

# Load Button
load_btn = tk.Button(root, text="Load CSV File", command=load_csv)
load_btn.pack(pady=5)

# Signal Type
signal_type = tk.StringVar()
signal_menu = ttk.Combobox(root, textvariable=signal_type)
signal_menu['values'] = ("ECG", "Temperature", "Respiration")
signal_menu.set("Select Signal Type")
signal_menu.pack(pady=5)

# Filter Buttons

filter_frame = tk.Frame(root)
filter_frame.pack(pady=10)

lpf_btn = tk.Button(filter_frame, text="Low Pass Filter", command=apply_lpf)
lpf_btn.grid(row=0,column=0,padx=5)

hpf_btn = tk.Button(filter_frame, text="High Pass Filter", command=apply_hpf)
hpf_btn.grid(row=0,column=1,padx=5)

bpf_btn = tk.Button(filter_frame, text="Band Pass Filter", command=apply_bpf)
bpf_btn.grid(row=0,column=2,padx=5)

# FFT Button
fft_btn = tk.Button(root, text="Show FFT", command=show_fft)
fft_btn.pack(pady=10)

# Reset
reset_btn = tk.Button(root, text="Reset Signal", command=reset_signal)
reset_btn.pack(pady=5)

# Statistics Display

stats_text = tk.StringVar()
stats_label = tk.Label(root, textvariable=stats_text, font=("Arial",12), justify="left")
stats_label.pack(pady=20)

root.mainloop()