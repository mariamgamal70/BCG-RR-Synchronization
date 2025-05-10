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
