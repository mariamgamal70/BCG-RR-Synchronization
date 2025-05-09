import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
def calculate_errors(bcg_heartrate, aligned_rr_hr, bcg_heartrate_timestamps):
    # Plot the aligned RR heart rate and BCG heart rate
    plt.figure(figsize=(12, 6))
    plt.plot(bcg_heartrate_timestamps, bcg_heartrate, label="BCG Heart Rate (BPM)", color="orange", linewidth=2)
    plt.plot(bcg_heartrate_timestamps, aligned_rr_hr, label="RR Heart Rate (BPM)", color="blue", linestyle="--", linewidth=2)
    plt.xlabel("Time (ms)")
    plt.ylabel("Heart Rate (BPM)")
    plt.title("Comparison of BCG and RR Heart Rates")
    plt.legend()
    plt.grid()
    plt.savefig("heart_rate_comparison_plot.png", dpi=300) 

    #error calculation and plotting 
    # Compute error metrics
    # Calculate MAE (Mean Absolute Error)
    mae = mean_absolute_error(aligned_rr_hr, bcg_heartrate)
    # Calculate RMSE (Root Mean Square Error)
    rmse = np.sqrt(mean_squared_error(aligned_rr_hr, bcg_heartrate))
    # Calculate MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((aligned_rr_hr - bcg_heartrate) / aligned_rr_hr)) * 100

    # Print error metrics``
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Root Mean Square Error (RMSE): {rmse:.2f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

    # Bland-Altman plot
    mean_values = (bcg_heartrate + aligned_rr_hr) / 2
    differences = bcg_heartrate - aligned_rr_hr
    plt.figure(figsize=(10, 6))
    plt.scatter(mean_values, differences, alpha=0.5)
    plt.axhline(np.mean(differences), color='red', linestyle='--', label="Mean Difference")
    plt.axhline(np.mean(differences) + 1.96 * np.std(differences), color='blue', linestyle='--', label="Upper Limit (95%)")
    plt.axhline(np.mean(differences) - 1.96 * np.std(differences), color='blue', linestyle='--', label="Lower Limit (95%)")
    plt.xlabel("Mean of BCG and RR Heart Rates (BPM)")
    plt.ylabel("Difference (BCG - RR) (BPM)")
    plt.title("Bland-Altman Plot")
    plt.legend()
    plt.grid()
    plt.savefig("bland_altman_plot.png", dpi=300)
    plt.close()
    # Pearson correlation
    correlation = np.corrcoef(bcg_heartrate, aligned_rr_hr)[0, 1]

    # Scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(aligned_rr_hr, bcg_heartrate, alpha=0.5)
    plt.xlabel("RR Heart Rate (BPM)")
    plt.ylabel("BCG Heart Rate (BPM)")
    plt.title(f"Pearson Correlation Plot (r = {correlation:.2f})")
    plt.grid()
    plt.savefig("pearson_scatter_plot.png", dpi=300)
    plt.close()
    # Boxplot for BCG and RR heart rates
    plt.figure(figsize=(8, 6))
    plt.boxplot([bcg_heartrate, aligned_rr_hr], labels=["BCG Heart Rate", "RR Heart Rate"], patch_artist=True, 
                boxprops=dict(facecolor="lightblue", color="blue"), medianprops=dict(color="red"))
    plt.ylabel("Heart Rate (BPM)")
    plt.title("Boxplot of Estimated BCG Heart Rate and Reference RR Heart Rate")
    plt.grid(axis="y")
    plt.savefig("heart_rate_boxplot.png", dpi=300)
    plt.close()
    