import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, windows, welch, detrend
from matplotlib.ticker import ScalarFormatter

from FFT_Analysis_Functions import fileReader, read_window, extract_impacts, apply_hamming_correction, single_fft_analysis


filePathVibe = './DATA/ENLIGHT/LaRC_Data/LaRC_test1_run0008.txt'
filePathVibe = './DATA/ENLIGHT/LaRC_Data/LaRC_test2_run0010.txt'
filePathVibe = './DATA/ENLIGHT/LaRC_Data/LaRC_test3_run0012_bad.txt'
filePathVibe = './DATA/ENLIGHT/LaRC_Data/LaRC_test4_run0022_noKuliteData.txt'
filePathVibe = './DATA/ENLIGHT/LaRC_Data/LaRC_test5_run0024.txt'
filePathVibe = './DATA/ENLIGHT/LaRC_Data/LaRC_test6_run0026.txt'
filePathVibe = './DATA/ENLIGHT/LaRC_Data/LaRC_test7_run0028.txt'
filePathVibe = './DATA/ENLIGHT/LaRC_Data/LaRC_test8_run0030.txt'
filePathVibe = './DATA/ENLIGHT/LaRC_Data/LaRC_test9_run0032.txt'
filePathVibe = './DATA/ENLIGHT/LaRC_Data/LaRC_test10_run0034.txt'
filePathVibe = './DATA/ENLIGHT/LaRC_Data/LaRC_test11_run0036.txt'
filePathVibe = './DATA/ENLIGHT/LaRC_Data/LaRC_test12_run0038.txt'
filePathVibe = './DATA/ENLIGHT/LaRC_Data/LaRC_test13_run0040.txt'
filePathVibe = './DATA/ENLIGHT/LaRC_Data/LaRC_test14_run0042.txt'
filePathVibe = './DATA/ENLIGHT/LaRC_Data/LaRC_test15_run0044_5secIRdata.txt'
filePathVibe = './DATA/ENLIGHT/LaRC_Data/LaRC_test16_run046_noKuliteData.txt'
filePathVibe = './DATA/ENLIGHT/LaRC_Data/LaRC_test17_run0048.txt'
filePathVibe = './DATA/ENLIGHT/LaRC_Data/LaRC_test18_run0050.txt'
filePathVibe = './DATA/ENLIGHT/LaRC_Data/LaRC_test19_run0052_noIRdata.txt'

filePathVibe = './DATA/ENLIGHT/LaRC_Data/LaRC_test8_run0030.txt'
#filePathTemp = './DATA/ENLIGHT/Peaks.20250325154942_TempTest.txt'

import os

# Save location: your system's Downloads folder (cross-platform)
downloads_path = os.path.join(os.path.expanduser("~"), "Downloads")



dfVibe, timeSpanV, samplingFreqV = fileReader(filePathVibe, False, True, start_trim=125)

startTime = 125;

for i in range(0, 25):
    currentTime = startTime + i   
    sensorWL_window, time_window = read_window(filePathVibe, start_sec=currentTime, end_sec=currentTime+1)

    print(sensorWL_window)
    print(time_window)

    # Assuming df has 5 columns: df.columns = ['A', 'B', 'C', 'D', 'E']
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 10))  # 3x2 layout
    axes = axes.flatten()  # Flatten the 2D axes array for easy iteration
    fig.suptitle(f"Signal Detrending for Time Snip at t = {currentTime} to {currentTime+1}", fontsize=16)

    for i, col in enumerate(sensorWL_window.columns):
        axes[i].plot(detrend(sensorWL_window[col]))
        axes[i].set_title(col)

    # Hide the unused 6th subplot
    if len(sensorWL_window.columns) < 6:
        axes[-1].axis('off')

    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(12, 10))  # 3x2 layout
    ax = ax.flatten()  # Flatten to make it easier to iterate
    fig.suptitle("Frequency Analysis per sensor output", fontsize=16)


    # for i, col in enumerate(sensorWL_window.columns):

    #     detrended_signal = detrend(sensorWL_window[col])
    #     positive_freqs, positive_power = single_fft_analysis(detrended_signal, time_window)

    #     ax[i].plot(positive_freqs, positive_power)
    #     ax[i].set_title(col)
    #     ax[i].set_xlabel("Frequency (Hz)")
    #     ax[i].set_ylabel("Power")

    # for j in range(len(sensorWL_window.columns), len(ax)):
    #     ax[j].axis('off')

    # plt.tight_layout()

    # Construct full filename
    filename = f"FFT_Sensor_Plots_t{currentTime}s.png"
    full_path = os.path.join(downloads_path, filename)
    # Save the figure
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {full_path}")


plt.show()

