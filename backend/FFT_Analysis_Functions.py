import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, windows, welch
from matplotlib.ticker import ScalarFormatter


def fileReader(filePath, flip, plot):
    
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

    if plot:

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
            # Disable scientific notation
            formatter = ScalarFormatter(useMathText=False)
            formatter.set_scientific(False)
            formatter.set_useOffset(False)

            ax.yaxis.set_major_formatter(formatter)

        

        # Optional: hide unused subplot (bottom right one)
        if dims[1] < 6:
            axis[2, 1].axis('off')

        plt.tight_layout()

    #print(sensorWL.head())
    #print(time_in_seconds.head())
    return sensorWL, time_in_seconds, samplingFreq

def extract_impacts(data, time, window_sec=0.2, n_peaks=10):
    """
    Extracts n_peaks impact windows from 1D data using automatic peak detection.

    Parameters:
        data (pd.Series or np.array): Sensor data (e.g., wavelength)
        time (pd.Series or np.array): Corresponding time values
        window_sec (float): Duration of window (in seconds)
        n_peaks (int): Number of peaks (impacts) to extract

    Returns:
        np.array: Array of shape (n_window_samples, n_peaks)
    """
    # Convert to numpy arrays
    data = np.array(data)
    time = np.array(time)

    # Compute sampling rate from time vector
    dt = np.mean(np.diff(time))
    sampling_rate = 1 / dt

    # Find peaks in the data
    peaks, _ = find_peaks(np.abs(data), prominence=0.01)  # Adjust prominence as needed

    if len(peaks) < n_peaks:
        raise ValueError(f"Only found {len(peaks)} peaks, but n_peaks={n_peaks} requested.")

    # Select top n_peaks by prominence
    prominences = np.abs(data[peaks])
    top_indices = np.argsort(prominences)[-n_peaks:]
    selected_peaks = np.sort(peaks[top_indices])  # ensure chronological order

    # Convert window size to number of samples
    half_window = int((window_sec / 2) * sampling_rate)
    window_length = 2 * half_window

    impacts_matrix = []

    for peak_idx in selected_peaks:
        start = max(0, int(peak_idx - 0.25 * window_length))
        end = min(len(data), int(peak_idx + 0.75 * window_length))

        if end - start == window_length:
            impacts_matrix.append(data[start:end])

    # Stack into matrix: shape (n_window_samples, n_peaks)
    impacts_matrix = np.column_stack(impacts_matrix)

    window_time = np.arange(window_length) * dt

    return impacts_matrix, window_time


def apply_hamming_correction(signal, time) -> np.ndarray:
    """
    Applies Hamming window correction to a 1D signal.
    """
    signal = np.array(signal)
    time = np.array(time)

    n_samples = len(signal)
    n = np.arange(n_samples)
    hamming_window = 0.54 - 0.46 * np.cos(2 * np.pi * n / (n_samples - 1))


    k1 = np.sum(hamming_window * signal) / np.sum(hamming_window)
    k2 = np.sqrt(n_samples / np.sum(hamming_window ** 2))

    corrected_signal = hamming_window * (signal - k1) * k2
    return corrected_signal


def single_fft_analysis(signal: np.ndarray, time: np.ndarray):
    """
    Applies FFT to a corrected signal and plots the result.
    """
    signal = np.array(signal)
    time = np.array(time)

    #corrected_signal = apply_hamming_correction(signal, time)
    n = len(signal)
    fs = n / (time[-1] - time[0])
    freq_vector = np.fft.fftfreq(n, d=1/fs)
    fft_result = np.fft.fft(signal)
    power_spectrum = np.abs(fft_result) ** 2

    # Keep only the positive frequencies
    positive_freqs = freq_vector[:n // 2]
    positive_power = power_spectrum[:n // 2]

    return positive_freqs, positive_power

