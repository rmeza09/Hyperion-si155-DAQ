import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


def fileReader(filePath, flip):
    
    df = pd.read_csv(filePath, skiprows=45, sep='\s+')

    #print(df.head())
    date = df.iloc[0, 0]
    sensorWL = df.iloc[:, -5:]
    dims = sensorWL.shape
    time = pd.to_datetime(df.iloc[:, 1], format='%H:%M:%S.%f')

    # Convert the datetime to seconds since the start of the time series
    time_in_seconds = (time - time.iloc[0]).dt.total_seconds()

    # Replace the original time variable with the numerical version
    timespan = time_in_seconds.iloc[-1] - time_in_seconds.iloc[0]
    samplingFreq = dims[0] / timespan

    print('File Name: ', filePath[15:])
    print('Date Acquired: ', date)
    print('Sampling Frequency: ', samplingFreq)
    print("Time Span: ", timespan)
    print('Number of Samples: ', dims[0], '\n')

    fig, axis = plt.subplots(3, 2, figsize=(12, 8))  # 3 rows, 2 columns

    for i in range(0, dims[1]):
        row = i // 2
        col = i % 2
        data = sensorWL.iloc[:, i]

        if i % 2 == 1 and flip:
            data *= -1

        ax = axis[row, col]
        ax.plot(time_in_seconds, data)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Center Wavelength (nm)')
        ax.set_title('Wavelength Change vs Time')
        ax.grid(True)

    # Optional: hide unused subplot (bottom right one)
    if dims[1] < 6:
        axis[2, 1].axis('off')

    plt.tight_layout()

    #print(sensorWL.head())
    #print(time_in_seconds.head())
    return sensorWL, timespan, samplingFreq

def extract_impacts(signal, time, sampling_rate, n_peaks=10, window_sec=0.2):
    """
    Extracts n_peaks impact windows from a 1D signal column.
    
    Parameters:
        signal (np.array): 1D array of the signal (wavelengths)
        time (np.array): 1D array of time values (same length as signal)
        sampling_rate (float): samples per second (Hz)
        n_peaks (int): number of impact peaks to find
        window_sec (float): total window duration (centered around peak)
    
    Returns:
        impacts_matrix (np.array): shape (n_window_samples, n_peaks)
    """
    # Convert to numpy arrays (in case passed as pandas series)
    signal = np.array(signal)
    time = np.array(time)

    # Find peaks (adjust prominence to isolate big impacts)
    peaks, _ = find_peaks(signal, prominence=0.01, distance=sampling_rate * 0.05)
    
    # Sort peaks by prominence and pick the top N
    prominences = np.abs(signal[peaks])
    top_indices = np.argsort(prominences)[-n_peaks:]
    selected_peaks = np.sort(peaks[top_indices])  # sort by time

    # Window size in samples
    half_window = int((window_sec / 2) * sampling_rate)
    window_length = 2 * half_window

    impacts_matrix = []

    for peak_idx in selected_peaks:
        start = max(0, peak_idx - half_window)
        end = min(len(signal), peak_idx + half_window)

        # Ensure we get the same window length for all
        if end - start == window_length:
            window = signal[start:end]
            impacts_matrix.append(window)

    # Stack into a matrix (n_window_samples, 10)
    impacts_matrix = np.column_stack(impacts_matrix)

    return impacts_matrix


filePathVibe = './DATA/ENLIGHT/Peaks.20250325154128_VibeTest.txt'
filePathTemp = './DATA/ENLIGHT/Peaks.20250325154942_TempTest.txt'

dfVibe, timeSpanV, samplingFreqV = fileReader(filePathVibe, False)
dfTemp, timeSpanT, samplingFreqT = fileReader(filePathTemp, True)

# Extract impact windows
impacts_vibe = extract_impacts(dfVibe, timeSpanV, samplingFreqV)
print(impacts_vibe.shape)

plt.show()