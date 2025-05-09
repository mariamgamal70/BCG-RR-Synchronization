import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from band_pass_filtering import band_pass_filtering
from compute_vitals import vitals
from scipy.signal import resample_poly
from detect_body_movements import detect_patterns
from modwt_matlab_fft import modwt
from modwt_mra_matlab_fft import modwtmra
# === 1. Load CSV ===
file_path = "C:/Users/maria/OneDrive/Desktop/HEM year 4 term 1/data analysis/final project/dataset/dataset/data/03/BCG/03_20231103_BCG.csv"  # Update with your filename
df = pd.read_csv(file_path, header=None, skiprows=1).values

bcg_signal = np.array(df[:, 0]) 
fs = 140  # Hz
start_timestamp = np.array(df[0, 1])   # convert ms to seconds

# === 2. Generate timestamps ===
time_axis = np.arange(len(bcg_signal)) / fs
timestamps = start_timestamp + time_axis

filtered_bcg, timestamps = detect_patterns(0, fs*10, fs*10, bcg_signal, timestamps, plot=0)
# === 3. Bandpass filter (0.7–3.5 Hz) ===
filtered_bcg = band_pass_filtering(filtered_bcg, fs, filter_type="bcg")

fs_down = 50
down_bcg = resample_poly(filtered_bcg, up=5, down=14)
time_ms_down = np.arange(len(down_bcg)) * (1000 / fs_down)
w = modwt(down_bcg, 'bior3.9', 4)
dc = modwtmra(w, 'bior3.9')
wavelet_cycle = dc[4]
win_duration_sec = 10
win_size = int(win_duration_sec * fs_down)
window_limit = int(len(down_bcg) / win_size)
mpd = int(0.5 * fs_down) 
# 4. Detect heart rate from each window
t1 = 0
t2 = win_size

heart_rates = vitals(t1, t2, win_size, window_limit, wavelet_cycle, time_ms_down, mpd)
print('Heart Rate Information')
print('Minimum pulse : ', np.around(np.min(heart_rates)))
print('Maximum pulse : ', np.around(np.max(heart_rates)))
print('Average pulse : ', np.around(np.mean(heart_rates)))
# 5. Plot heart rate over time
plt.plot(np.arange(len(heart_rates)) * win_duration_sec, heart_rates, marker='o')
plt.xlabel("Time (s)")
plt.ylabel("Heart Rate (bpm)")
plt.title("Heart Rate Over Time from BCG")
plt.grid(True)
plt.savefig("bcg_signal_plot5.png", dpi=300) 