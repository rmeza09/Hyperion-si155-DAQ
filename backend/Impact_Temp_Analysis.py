
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, windows, welch
from matplotlib.ticker import ScalarFormatter

from FFT_Analysis_Functions import fileReader, extract_impacts, apply_hamming_correction, single_fft_analysis


def processFBGData(dfVibe, timeSpanV, samplingFreqV):
    #print(dfVibe.iloc[:, 0])
    test = np.array(dfVibe.iloc[:, 0])
    #testtime = np.array(timeSpanV.iloc[:, 0])
    #print(test)
    #print(timeSpanV)

    # Extract impact windows
    #impacts_vibe = extract_impacts(dfVibe.iloc[:, 0], timeSpanV)
    #print(impacts_vibe.shape)
    numSensors = 5
    allCollectedPeaks = np.zeros((numSensors*10, 6))  # Initialize a 2D array to store the collected peaks

    for k in range(0, numSensors):

        impacts_vibe, window_time = extract_impacts(dfVibe.iloc[:,k], timeSpanV)
        # Initialize a list to store corrected signals
        corrected = []
        # Apply Hamming correction for each impact and store the result
        for j in range(impacts_vibe.shape[1]):
            correctedSig = apply_hamming_correction(impacts_vibe[:, j], window_time)
            corrected.append(correctedSig)
        # Stack the corrected signals into a 2D array (shape: [1000, 10])
        correctedSignal = np.column_stack(corrected)
        #print(correctedSignal.shape)  # Should print (1000, 10)

        # Plot impact windows
        fig, axis = plt.subplots(2, 5, figsize=(18, 6))
        sensor_wavelength = np.round(impacts_vibe[0, 0]).astype(int)
        fig.suptitle(f"Center Wavelength: {sensor_wavelength}nm", fontsize=16)
        for i in range(impacts_vibe.shape[1]):
            row = i // 5
            col = i % 5
            axis[row, col].plot(correctedSignal[:, i]) # swap with impacts_vibe if you want to see the original signal
            axis[row, col].set_title(f"Impact {i+1}")
            axis[row, col].set_xlabel("Sample")
            axis[row, col].set_ylabel("Wavelength (nm)")
            axis[row, col].grid(True)
            # Disable scientific notation
            formatter = ScalarFormatter(useMathText=False)
            formatter.set_scientific(False)
            formatter.set_useOffset(False)
            ax = axis[row, col]
            ax.yaxis.set_major_formatter(formatter)
        plt.tight_layout()

        # Compute FFT for each corrected signal
        fft_power = []
        fft_freq = []
        for i in range(correctedSignal.shape[1]):
            freq, power = single_fft_analysis(correctedSignal[:, i], window_time)
            fft_power.append(power)
            fft_freq.append(freq)
        # Stack the FFT results into a 2D array (shape: [500, 10])
        fft_power = np.column_stack(fft_power)
        fft_freq = np.column_stack(fft_freq)

        extractedPeaks = []
        for p in range(fft_power.shape[1]):  # Iterate over each column (impact)
            peaks, _ = find_peaks(fft_power[:, p], prominence=0.3, distance=25)  # Find peaks in the power spectrum
            if len(peaks) < 6:
                peaks = np.pad(peaks, (0, 6 - len(peaks)), constant_values=0)
            extractedPeaks.append(fft_freq[peaks[:6], p])  # Match peaks to frequencies
        extractedPeaks = np.column_stack(extractedPeaks).T  # Stack the extracted peaks into a 2D array
        #print(extractedPeaks.shape)

        start_row = k * 10
        end_row = start_row + 10
        allCollectedPeaks[start_row:end_row, :] = extractedPeaks


        # Create a figure and axis (we'll hide the axis)
        fig, ax = plt.subplots()
        ax.axis('tight')
        ax.axis('off')

        # set column and row labels
        #columns = [f"Num: {i+1}" for i in range(extractedPeaks.shape[1])]
        rows = [f"PS Analysis: {i+1}" for i in range(extractedPeaks.shape[0])]

        # Create the table
        table = ax.table(
            cellText=np.round(extractedPeaks, 2),
            rowLabels=rows,
            #colLabels=columns,
            cellLoc='center',
            loc='center'
        )

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)

        plt.tight_layout()


        
        #print(fft_power.shape)
        #print(fft_freq.shape)

        #print(fft_freq)

        fig, axis = plt.subplots(2, 5, figsize=(18, 6))
        sensor_wavelength = np.round(impacts_vibe[0, 0]).astype(int)
        fig.suptitle(f"Center Wavelength: {sensor_wavelength}nm", fontsize=16)
        for m in range(fft_power.shape[1]):
            row = m // 5
            col = m % 5
            ax = axis[row, col]
            axis[row, col].plot(fft_freq[:, m], fft_power[:, m]) 
            axis[row, col].set_title(f"Power Spectrum {m+1}")
            axis[row, col].set_xlabel("Frequency (Hz)")
            axis[row, col].set_ylabel("Power")
            axis[row, col].grid(True)

            peaks, _ = find_peaks(fft_power[:, m], prominence=0.3, distance=25)
            selected_peaks = peaks[:6]  # Select up to the first 6 peaks
            ax.scatter(fft_freq[selected_peaks, m], fft_power[selected_peaks, m], color='red', label="Reported Peaks")

            # Disable scientific notation
            formatter = ScalarFormatter(useMathText=False)
            formatter.set_scientific(False)
            formatter.set_useOffset(False)
            ax.yaxis.set_major_formatter(formatter)
        plt.tight_layout()

    #print(allCollectedPeaks)

    # Save all collected peaks to CSV for Excel use
    output_path = "./DATA/CSV/allCollectedPeaks.csv"
    np.savetxt(output_path, allCollectedPeaks, delimiter=",", fmt="%.2f")
    print(f"Saved all collected peaks to '{output_path}'")



    plt.show()


#filePathVibe = './DATA/ENLIGHT/Peaks.20250325154128_VibeTest.txt' # OG center impacts zone 5
#filePathVibe = './DATA/ENLIGHT/Peaks.20250409150553.txt' # Center impacts zone 5
# filePathVibe = './DATA/ENLIGHT/Peaks.20250409151137.txt' # Bottom Left corner impacts zone 7
# filePathVibe = './DATA/ENLIGHT/Peaks.20250409151742.txt' # Mid-line left impacts zone 4
filePathVibe = './DATA/ENLIGHT/Peaks.20250320140346.txt' # Bottom Center-line impacts zone 8
#filePathTemp = './DATA/ENLIGHT/Peaks.20250325154942_TempTest.txt'

dfVibe, timeSpanV, samplingFreqV = fileReader(filePathVibe, False, True)
#dfTemp, timeSpanT, samplingFreqT = fileReader(filePathTemp, True, True)

processFBGData(dfVibe, timeSpanV, samplingFreqV)