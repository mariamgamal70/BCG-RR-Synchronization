# BCG-RR Synchronization and Heart Rate Analysis

This project processes **Ballistocardiogram (BCG)** and **R-R interval (RR)** data to estimate heart rate, synchronize timestamps, and calculate errors between the two datasets. It includes signal processing, body movement detection, filtering, downsampling, and wavelet decomposition to extract meaningful insights.

## ✨ Features

✅ **BCG and RR Data Synchronization**: Aligns timestamps and signals from BCG and RR datasets.  
✅ **Body Movement Detection**: Removes segments of the BCG signal affected by body movements.  
✅ **Signal Filtering**: Applies band-pass filtering to isolate heartbeat and breathing signals.  
✅ **Downsampling**: Reduces the sampling frequency of the BCG signal to 50 Hz.  
✅ **Wavelet Decomposition**: Decomposes the BCG signal into components for heartbeat analysis.  
✅ **Heart Rate Estimation**: Computes heart rate from the processed BCG signal.  
✅ **Error Calculation**: Compares BCG-derived heart rate with RR-derived heart rate and calculates errors.

---

## 📝 Prerequisites

- Python **3.10** or higher
- Required Python libraries:
  - `numpy`
  - `pandas`
  - `scipy`
  - `scikit-learn`

Install dependencies using:

```bash
pip install numpy pandas scipy scikit-learn
```
## 📁 File Structure
File	Description
main.py:	Main script that processes data and runs pipeline
band_pass_filtering.py:	Band-pass filter implementation
compute_vitals.py:	Computes heart rate from wavelet cycles
detect_body_movements.py:	Detects and removes body movements in BCG
modwt_matlab_fft.py:	Wavelet decomposition function
modwt_mra_matlab_fft.py:	Wavelet multi-resolution analysis
error_calculations.py:	Calculates and visualizes errors

## 🚀 How to Run
Place the BCG and RR data files in the following directories:
```bash
dataset/dataset/data/<subject_id>/BCG/*.csv
dataset/dataset/data/<subject_id>/Reference/RR/*.csv

Run the main script:
```bash
python main.py
The script will process each subject’s data, synchronize timestamps, estimate heart rates, and calculate errors.

## 🗃️ Input Data Format
BCG Data
A CSV file with 2 columns:

BCG signal values

Start time (Unix timestamp in milliseconds)

RR Data
A CSV file with 2 columns:

Timestamps (YYYY/MM/DD HH:MM:SS format)

Heart rate (BPM)

## 📤 Output
Heart Rate Metrics:

Minimum, maximum, and average heart rates for both BCG and RR datasets.

Error Metrics:

Error statistics between BCG-derived and RR-derived heart rates.

Plots:

Visualizations of heart rate comparisons and error distributions (if implemented in error_calculations.py)

## 🔑 Key Functions
Function	Description
detect_patterns	Detects body movements and removes affected segments
band_pass_filtering	Applies Chebyshev Type I band-pass filter
modwt, modwtmra	Wavelet decomposition and multi-resolution analysis
vitals	Estimates heart rate from wavelet cycles
calculate_errors	Computes and visualizes errors between heart rates
