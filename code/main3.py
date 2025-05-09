# Import required libraries
import math
import os
from datetime import datetime, timezone, timedelta
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import glob
from scipy.signal import resample_poly

from band_pass_filtering import band_pass_filtering
from compute_vitals import vitals
from detect_body_movements import detect_patterns
from modwt_matlab_fft import modwt
from modwt_mra_matlab_fft import modwtmra
from data_subplot import data_subplot

dataset_root = '../../dataset/dataset/data/'

for subject_id in os.listdir(dataset_root):
    subject_path = os.path.join(dataset_root, subject_id)
    print (f'\nProcessing subject: {subject_id}')
     # Get all RR files
    rr_path = os.path.join(subject_path, "Reference/RR", "*.csv")
    rr_files = glob.glob(rr_path)
    
    if len(rr_files) == 0:
        print('No RR files found. Skipping RR processing...')
        continue

    # Extract the base names of RR files (without "_RR.csv")
    rr_base_names = [os.path.basename(rr_file).replace("_RR.csv", "") for rr_file in rr_files]
    
    # Get all BCG files
    bcg_path = os.path.join(subject_path, "BCG", "*.csv")
    bcg_files = glob.glob(bcg_path)
    
    # Filter BCG files to include only those with matching RR files
    matching_bcg_files = [
        bcg_file for bcg_file in bcg_files
        if os.path.basename(bcg_file).replace("_BCG.csv", "") in rr_base_names
    ]

    if len(matching_bcg_files) == 0:
        print('No matching BCG files found. Skipping BCG processing...')
        continue
    
    # Load RR data (heart rate)
    for rr_file, bcg_file in zip(rr_files, matching_bcg_files):
        print('\nLoading RR data')
        rr_data = pd.read_csv(rr_file, header=None, skiprows=1).values
        rr_timestamps = np.array([
            datetime.strptime(ts, "%Y/%m/%d %H:%M:%S").replace(tzinfo=timezone(timedelta(hours=8))).astimezone(timezone.utc).timestamp()
            for ts in rr_data[:, 0]
        ])
        rr_timestamps *= 1000 # Convert RR timestamps to milliseconds
        heart_rate = np.array(rr_data[:, 1])     # Heart rate (BPM)
        heart_rate = heart_rate.astype(float)  # Convert to float
        rr_intervals = rr_data[:, 2]   # RR intervals (seconds)
        print ('Heart Rate Information')
        print('Minimum heart rate : ', np.around(np.min(heart_rate)))
        print('Maximum heart rate : ', np.around(np.max(heart_rate)))
        print('Average heart rate : ', np.around(np.mean(heart_rate)))

#==========================================================================================
    # Load BCG data
        print('\nLoading BCG data')
        bcg_dataset = pd.read_csv(bcg_file, header=None, skiprows=1).values
        bcg_raw_signal = np.array(bcg_dataset[:, 0])       # BCG values
        bcg_start_time = np.array(bcg_dataset[0, 1])   # Start time (UTC)
        bcg_fs = 140          # Sampling frequency (140 Hz)

        # Generate timestamps for each sample
        bcg_timestamps = np.arange(bcg_raw_signal.size) * (1000/bcg_fs) + bcg_start_time

        WINDOW_TIME_SEC= 10  # seconds
        FS=50

        # Process BCG signal

        # Detects body movements (e.g., rolling over) in the BCG signal and removes them.
        clean_bcg, clean_bcg_time = detect_patterns(0, bcg_fs*WINDOW_TIME_SEC, bcg_fs*WINDOW_TIME_SEC, bcg_raw_signal, bcg_timestamps, plot=0)

        # Isolates BCG (heartbeat) and breathing signals using frequency filters.
        clean_bcg = band_pass_filtering(clean_bcg, bcg_fs, "bcg")  

        #downsample the BCG signal to 50 Hz
        downsampled_clean_bcg = resample_poly(clean_bcg, up=5, down=14)
        downsampled_clean_bcg_time = np.arange(len(downsampled_clean_bcg)) * (1000 / FS) +  bcg_start_time
        downsampled_clean_bcg_time
        # Wavelet transforms to decompose BCG into heartbeat components.    
        w = modwt(downsampled_clean_bcg, 'bior3.9', 4)
        dc = modwtmra(w, 'bior3.9')
        wavelet_cycle = dc[4]

        win_size = int(WINDOW_TIME_SEC * FS)
        limit = int(len(downsampled_clean_bcg) / win_size)
        t1 = 0
        t2 = win_size
        mpd = 1 # Minimum pulse duration (in seconds)

        #Estimate heart rate (from wavelet cycles) 
        beats = vitals(t1, t2, win_size, limit, wavelet_cycle, downsampled_clean_bcg_time, mpd)
        print('Heart Rate Information')
        print('Minimum pulse(BCG) : ', np.around(np.min(beats)))
        print('Maximum pulse(BCG) : ', np.around(np.max(beats)))
        print('Average pulse(BCG) : ', np.around(np.mean(beats)))
        
#==========================================================================================
        # Synchronize heart rate with RR data
        aligned_rr_hr = np.interp(downsampled_clean_bcg_time, rr_timestamps, heart_rate)

        # create new timestamps for aligned beats
        beats_time = downsampled_clean_bcg_time[:len(beats) * win_size:win_size] + (win_size / 2) * (1000 / FS)

        # Plot the synchronized heart rates
        plt.figure(figsize=(12, 6))
        plt.plot(beats_time, beats, label="BCG Heart Rate (BPM)", color="orange")
        plt.plot(downsampled_clean_bcg_time, aligned_rr_hr, label="RR Heart Rate (BPM)", color="blue", linestyle="--")
        plt.xlabel("Time (ms)")
        plt.ylabel("Heart Rate (BPM)")
        plt.title("Comparison of BCG and RR Heart Rates")
        plt.legend()
        plt.grid()
        plt.savefig("synchronized_heart_rate_plot.png", dpi=300)  # Save the plot as an 
print('\nEnd processing ...')