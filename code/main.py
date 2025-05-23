# Import required libraries
import math
import os
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import glob
from scipy.signal import resample_poly

from band_pass_filtering import band_pass_filtering
from compute_vitals import vitals
from detect_body_movements import detect_patterns
from modwt_matlab_fft import modwt
from modwt_mra_matlab_fft import modwtmra
from error_calculations import calculate_errors
dataset_root = '../../dataset/dataset/data/'

RR_TOTAL_HEARTRATE=[]
BCG_TOTAL_HEARTRATE=[]
RR_TOTAL_TIME=[]
BCG_TOTAL_TIME=[]

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
        rr_heart_rate = np.array(rr_data[:, 1])     # Heart rate (BPM)
        rr_heart_rate = rr_heart_rate.astype(float)  # Convert to float
        
        non_zero_mask = rr_heart_rate != 0
        rr_heart_rate = rr_heart_rate[non_zero_mask]
        rr_timestamps = rr_timestamps[non_zero_mask]
        
        WINDOW_TIME_SEC= 10  # seconds
        FS=50
        # Load BCG data
        print('\nLoading BCG data')
        bcg_dataset = pd.read_csv(bcg_file, header=None, skiprows=1).values
        bcg_raw_signal = np.array(bcg_dataset[:, 0])    # BCG values
        bcg_start_time = np.array(bcg_dataset[0, 1])    # Start time (UTC)
        bcg_fs = 140                                    # Sampling frequency (140 Hz)
        bcg_raw_timestamps = np.arange(bcg_raw_signal.size) * (1000/bcg_fs) + bcg_start_time
        
        # Detects body movements (e.g., rolling over) in the BCG signal and removes them.
        clean_bcg, clean_bcg_time = detect_patterns(0, bcg_fs*WINDOW_TIME_SEC, bcg_fs*WINDOW_TIME_SEC, bcg_raw_signal, bcg_raw_timestamps, plot=0)
        # Isolates BCG (heartbeat) and breathing signals using frequency filters.
        clean_bcg = band_pass_filtering(clean_bcg, bcg_fs, "bcg")  

        #downsample the BCG signal to 50 Hz
        clean_bcg = resample_poly(clean_bcg, up=5, down=14)
        # Generate timestamps for each downsampled sample
        clean_bcg_time = np.arange(clean_bcg.size) * (1000/FS) + bcg_start_time
        # Process BCG signal
        # Find the overlapping time range
        t1 = max(np.min(clean_bcg_time), np.min(rr_timestamps))
        t2 = min(np.max(clean_bcg_time), np.max(rr_timestamps))

        # Truncate BCG signal and timestamps
        bcg_mask = (clean_bcg_time >= t1) & (clean_bcg_time <= t2)
        synchronized_bcg_time = clean_bcg_time[bcg_mask]
        synchronized_bcg_signal = clean_bcg[bcg_mask]

        # Truncate ECG (RR) signal and timestamps
        rr_mask = (rr_timestamps >= t1) & (rr_timestamps <= t2)
        synchronized_rr_time = rr_timestamps[rr_mask]
        synchronized_rr_signal = rr_heart_rate[rr_mask]

        # Truncate ECG with BCG timestamps
        clean_RR_heart_rate,clean_RR_time = detect_patterns(0, FS*WINDOW_TIME_SEC, FS*WINDOW_TIME_SEC, synchronized_rr_signal, synchronized_rr_time, plot=0)
        # Wavelet transforms to decompose BCG into heartbeat components.    
        w = modwt(clean_bcg, 'bior3.9', 4)
        dc = modwtmra(w, 'bior3.9')
        wavelet_cycle = dc[4]

        win_size = int(WINDOW_TIME_SEC * FS)
        limit = int(len(clean_bcg) / win_size)
        t1 = 0
        t2 = win_size
        mpd = 1 # Minimum pulse duration (in seconds)

        #Estimate heart rate (from wavelet cycles) 
        bcg_heartrate = vitals(t1, t2, win_size, limit, wavelet_cycle, clean_bcg_time, mpd)
        #timeline of bcg heart rate to computed (average timeline of window size)
        bcg_heartrate_timestamps = clean_bcg_time[:len(bcg_heartrate) * win_size:win_size] + (win_size / 2) * (1000 / FS)
        print('Heart Rate Information (BCG)')
        print('Minimum pulse(BCG) : ', np.around(np.min(bcg_heartrate)))
        print('Maximum pulse(BCG) : ', np.around(np.max(bcg_heartrate)))
        print('Average pulse(BCG) : ', np.around(np.mean(bcg_heartrate)))
        #truncate the bigger size of heartrate to the smaller size of heartrate (interpolation)
        aligned_rr_hr = np.interp(bcg_heartrate_timestamps, clean_RR_time, clean_RR_heart_rate)
        print ('Heart Rate Information(RR)')
        print('Minimum heart rate(RR) : ', np.around(np.min(aligned_rr_hr)))
        print('Maximum heart rate(RR) : ', np.around(np.max(aligned_rr_hr)))
        print('Average heart rate(RR) : ', np.around(np.mean(aligned_rr_hr)))
        # Calculate errors and plot
        calculate_errors(bcg_heartrate, aligned_rr_hr, bcg_heartrate_timestamps)
        # Append results to the total lists
        RR_TOTAL_HEARTRATE.append(aligned_rr_hr)
        BCG_TOTAL_HEARTRATE.append(bcg_heartrate)
        RR_TOTAL_TIME.append(bcg_heartrate_timestamps)
        BCG_TOTAL_TIME.append(synchronized_bcg_time)
#==========================================================================================
calculate_errors(BCG_TOTAL_HEARTRATE, RR_TOTAL_HEARTRATE, BCG_TOTAL_TIME)

print('\nEnd processing ...')