import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, windows, welch, detrend, butter, filtfilt
from matplotlib.ticker import ScalarFormatter
import os


def fileReader(filePath, flip, plot, indivPlot, start_trim=0): 
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    df = pd.read_csv(filePath, skiprows=45, sep='\s+')

    date = df.iloc[0, 0]
    sensorWL = df.iloc[:, -5:]
    dims = sensorWL.shape
    time = pd.to_datetime(df.iloc[:, 1], format='%H:%M:%S.%f')

    # Convert to seconds
    time_in_seconds = (time - time.iloc[0]).dt.total_seconds()

    # Apply trimming
    keep_idx = time_in_seconds >= start_trim
    time_in_seconds = time_in_seconds[keep_idx].reset_index(drop=True)
    sensorWL = sensorWL[keep_idx.values].reset_index(drop=True)
    dims = sensorWL.shape  # Update dims after trimming

    timespan = time_in_seconds.iloc[-1] - time_in_seconds.iloc[0]
    samplingFreq = dims[0] / timespan if timespan > 0 else float('inf')

    print('File Name: ', filePath[15:])
    print('Date Acquired: ', date)
    print('Sampling Frequency: ', samplingFreq)
    print("Time Span: ", timespan)
    print('Number of Samples: ', dims[0], '\n')

    if plot:

        if indivPlot:
            for i in range(dims[1]):
                data = sensorWL.iloc[:, i]

                if i % 2 == 1 and flip:
                    data *= -1

                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(time_in_seconds, data)
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Center Wavelength (nm)')
                ax.set_title(f'Sensor {i+1}: Wavelength Change vs Time')
                ax.grid(True)

                formatter = ScalarFormatter(useMathText=False)
                formatter.set_scientific(False)
                formatter.set_useOffset(False)
                ax.yaxis.set_major_formatter(formatter)

                plt.tight_layout()
                

        else:
            fig, axis = plt.subplots(3, 2, figsize=(12, 8))  # 3 rows, 2 columns

            for i in range(dims[1]):
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
                formatter = ScalarFormatter(useMathText=False)
                formatter.set_scientific(False)
                formatter.set_useOffset(False)
                ax.yaxis.set_major_formatter(formatter)

            if dims[1] < 6:
                axis[2, 1].axis('off')

            plt.tight_layout()

    return sensorWL, time_in_seconds, samplingFreq

def read_window(filePath, start_sec, end_sec):
    import pandas as pd

    df = pd.read_csv(filePath, skiprows=45, sep='\s+')

    # Get the wavelength data (last 5 columns assumed)
    sensorWL = df.iloc[:, -5:]

    # Extract and convert time to seconds
    time = pd.to_datetime(df.iloc[:, 1], format='%H:%M:%S.%f')
    time_in_seconds = (time - time.iloc[0]).dt.total_seconds()

    # Select time window
    mask = (time_in_seconds >= start_sec) & (time_in_seconds <= end_sec)
    time_window = time_in_seconds[mask].reset_index(drop=True)
    sensorWL_window = sensorWL[mask.values].reset_index(drop=True)

    return sensorWL_window, time_window



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

def apply_highpass_filter(signal, cutoff, fs, order=4):
    """
    Apply a high-pass Butterworth filter to a signal.
    
    Parameters:
        signal (array-like): Input signal.
        cutoff (float): Cutoff frequency in Hz.
        fs (float): Sampling rate in Hz.
        order (int): Filter order (default = 4).
    
    Returns:
        filtered_signal (array-like): High-pass filtered signal.
    """
    nyq = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyq  # Normalize cutoff
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

def find_prominent_peaks(freqs, power, min_freq=0, max_freq=2500, min_prominence=0.01, top_n=5):
    """
    Identify prominent peaks in a power spectrum.
    
    Parameters:
        freqs (np.ndarray): Frequency array (Hz).
        power (np.ndarray): Power spectrum array.
        min_freq (float): Minimum frequency to consider.
        max_freq (float): Maximum frequency to consider.
        min_prominence (float): Minimum prominence of peaks.
        top_n (int): Number of top peaks to return.

    Returns:
        peak_freqs (np.ndarray): Frequencies of top peaks.
        peak_powers (np.ndarray): Power values of top peaks.
    """
    # Limit to desired frequency range
    mask = (freqs >= min_freq) & (freqs <= max_freq)
    freqs_range = freqs[mask]
    power_range = power[mask]
    
    # Find peaks with required prominence
    peaks, properties = find_peaks(power_range, prominence=min_prominence)
    
    # Sort peaks by height (descending)
    sorted_indices = peaks[np.argsort(power_range[peaks])[::-1]]
    
    # Select top N peaks
    top_indices = sorted_indices[:top_n]
    peak_freqs = freqs_range[top_indices]
    peak_powers = power_range[top_indices]

    return peak_freqs, peak_powers



# def signal_detrend_analysis(filePathVibe, startTime, timeWindow, cutoff, plotSegments=False, plotFFT=False, saveFigures=False):

#     # Save location: your system's Downloads folder (cross-platform)
#     downloads_path = os.path.join(os.path.expanduser("~"), "Downloads")
    
#     for i in range(0, timeWindow):
#         currentTime = startTime + i   
#         sensorWL_window, time_window = read_window(filePathVibe, start_sec=currentTime, end_sec=currentTime+1)

#         print(sensorWL_window)
#         print(time_window)

#         if plotSegments:
#             # Assuming df has 5 columns: df.columns = ['A', 'B', 'C', 'D', 'E']
#             fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 10))  # 3x2 layout
#             axes = axes.flatten()  # Flatten the 2D axes array for easy iteration
#             fig.suptitle(f"Signal Detrending for Time Snip at t = {currentTime} to {currentTime+1}", fontsize=16)

#             for i, col in enumerate(sensorWL_window.columns):
#                 axes[i].plot(detrend(sensorWL_window[col]))
#                 axes[i].set_title(col)

#             # Hide the unused 6th subplot
#             if len(sensorWL_window.columns) < 6:
#                 axes[-1].axis('off')

#         if plotFFT:
#             fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(12, 10))  # 3x2 layout
#             ax = ax.flatten()  # Flatten to make it easier to iterate
#             fig.suptitle("Frequency Analysis per sensor output", fontsize=16)

#             fs = 5000  # sampling frequency, adjust as needed
#             cutoff_freq = cutoff  # cutoff frequency for high-pass filter, adjust as needed
#             for i, col in enumerate(sensorWL_window.columns):

#                 detrended_signal = detrend(sensorWL_window[col])
#                 filtered_signal = apply_highpass_filter(detrended_signal, cutoff_freq, fs, order=4)
#                 positive_freqs, positive_power = welch(filtered_signal, fs=fs, nperseg=1024, noverlap=512)
#                 #positive_freqs, positive_power = single_fft_analysis(filtered_signal, time_window)
#                 peak_freqs, peak_powers = find_prominent_peaks(positive_freqs, positive_power, min_freq=0, max_freq=2500, min_prominence=0.15 * np.max(positive_power), top_n=6)
#                 print(f"Top peaks for {col}: {peak_freqs}, Powers: {peak_powers}")

#                 ax[i].plot(positive_freqs, positive_power)
#                 ax[i].plot(peak_freqs, peak_powers, 'ro', label='Peaks')
#                 ax[i].set_title(col)
#                 ax[i].set_xlabel("Frequency (Hz)")
#                 ax[i].set_ylabel("Power")
#                 #ax[i].set_yscale('log')

#             for j in range(len(sensorWL_window.columns), len(ax)):
#                 ax[j].axis('off')

#             plt.tight_layout()

#         if saveFigures:
#             # Construct full filename
#             filename = f"FFT_Sensor_Plots_t{currentTime}s.png"
#             full_path = os.path.join(downloads_path, filename)
#             # Save the figure
#             plt.savefig(full_path, dpi=300, bbox_inches='tight')
#             print(f"Figure saved to: {full_path}")


#     plt.show()


def signal_detrend_analysis(filePathVibe, startTime, timeWindow, cutoff, plotSegments=False, plotFFT=False, saveFreqs=False, saveFigures=False):
    
    downloads_path = os.path.join(os.path.expanduser("~"), "Downloads")
    
    peak_matrix = []
    sensor_labels = []

    for i in range(0, timeWindow):
        currentTime = startTime + i   
        sensorWL_window, time_window = read_window(filePathVibe, start_sec=currentTime, end_sec=currentTime+1)
        
        if not sensor_labels:
            sensor_labels = list(sensorWL_window.columns)

        print(sensorWL_window)
        print(time_window)

        if plotSegments:
            fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 10))
            axes = axes.flatten()
            fig.suptitle(f"Signal Detrending for Time Snip at t = {currentTime} to {currentTime+1}", fontsize=16)

            for i, col in enumerate(sensorWL_window.columns):
                axes[i].plot(detrend(sensorWL_window[col]))
                axes[i].set_title(col)

            if len(sensorWL_window.columns) < 6:
                axes[-1].axis('off')

        if plotFFT:
            fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(12, 10))
            ax = ax.flatten()
            fig.suptitle("Frequency Analysis per sensor output", fontsize=16)

        fs = 5000
        cutoff_freq = cutoff
        peak_row = []

        for j, col in enumerate(sensorWL_window.columns):
            original_signal = sensorWL_window[col]

            detrended_signal = detrend(original_signal)
            filtered_signal = apply_highpass_filter(detrended_signal, cutoff_freq, fs, order=4)
            positive_freqs, positive_power = welch(filtered_signal, fs=fs, nperseg=1024, noverlap=512)
            
            # <-- MODIFIED: Calculate stats from the FFT power spectrum
            signal_mean = np.mean(positive_power)
            signal_std = np.std(positive_power)
            
            # <-- NEW: Define the noise threshold
            noise_threshold = signal_mean + (3 * signal_std)

            # Find initial candidate peaks
            min_prom = 0.15 * np.max(positive_power)
            peak_freqs, peak_intesities = find_prominent_peaks(
                positive_freqs, positive_power,
                min_freq=0, max_freq=2500,
                min_prominence=min_prom,
                top_n=6 # Find the top 6 most prominent peaks initially
            )
            
            # <-- NEW: Filter peaks based on the noise threshold
            significant_mask = peak_intesities >= noise_threshold
            filtered_freqs = peak_freqs[significant_mask]
            filtered_intensities = peak_intesities[significant_mask]

            # <-- MODIFIED: Pad the *filtered* arrays to 6 values
            padded_freqs = np.pad(filtered_freqs, (0, 6 - len(filtered_freqs)), constant_values=np.nan)
            padded_intensities = np.pad(filtered_intensities, (0, 6 - len(filtered_intensities)), constant_values=np.nan)

            interleaved_data = [val for pair in zip(padded_freqs, padded_intensities) for val in pair]
            
            peak_row.extend(interleaved_data)

            # <-- MODIFIED: Append all three stats to the row
            peak_row.append(signal_mean)
            peak_row.append(signal_std)
            peak_row.append(noise_threshold)

            if plotFFT:
                ax[j].plot(positive_freqs, positive_power)
                # Plot a horizontal line for the noise threshold
                ax[j].axhline(y=noise_threshold, color='r', linestyle='--', label=f'Threshold ({noise_threshold:.2e})')
                # Plot only the significant peaks that passed the filter
                ax[j].plot(filtered_freqs, filtered_intensities, 'ro', label='Significant Peaks')
                ax[j].set_title(col)
                ax[j].set_xlabel("Frequency (Hz)")
                ax[j].set_ylabel("Power")
                ax[j].legend()


        peak_matrix.append(peak_row)

        if plotFFT:
            for k in range(len(sensorWL_window.columns), len(ax)):
                ax[k].axis('off')
            plt.tight_layout()
            if saveFigures:
                filename = f"FFT_Sensor_Plots_t{currentTime}s.png"
                full_path = os.path.join(downloads_path, filename)
                plt.savefig(full_path, dpi=300, bbox_inches='tight')
                print(f"Figure saved to: {full_path}")
            
    if saveFreqs:
        # <-- MODIFIED: Update column names to include the threshold
        col_names = []
        for label in sensor_labels:
            for i in range(6):
                col_names.append(f"{label}_Peak{i+1}_Freq")
                col_names.append(f"{label}_Peak{i+1}_Intensity")
            col_names.append(f"{label}_Mean")
            col_names.append(f"{label}_StdDev")
            col_names.append(f"{label}_Threshold")

        peak_df = pd.DataFrame(peak_matrix, columns=col_names)
        
        excel_filename = f"Peak_Analysis_Data_{startTime}s_to_{startTime + timeWindow}s.xlsx"
        excel_path = os.path.join(downloads_path, excel_filename)
        peak_df.to_excel(excel_path, index=False)
        print(f"Peak analysis data saved to: {excel_path}")

    plt.show()
    plt.close()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, detrend, butter, filtfilt

# Assuming you have a read_window function defined elsewhere
# def read_window(filePathVibe, start_sec, end_sec):
#     ...
#     return sensor_data, time_vector

# --- Helper Function to apply the filter ---
def apply_highpass_filter(signal, cutoff, fs, order=4):
    """
    Applies a high-pass Butterworth filter to a signal.

    Args:
        signal (np.array): The input signal.
        cutoff (float): The cutoff frequency in Hz.
        fs (int): The sampling frequency in Hz.
        order (int): The order of the filter.

    Returns:
        np.array: The filtered signal.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

# --- Main function, now with filtering ---
def create_spectrograms(filePathVibe, startTime, timeWindow, fs=5000, cutoff_freq=100):
    """
    Loads, detrends, and high-pass filters a raw signal, then
    generates a spectrogram for each sensor.
    """
    print(f"Loading {timeWindow} seconds of data starting from {startTime}s...")
    endTime = startTime + timeWindow
    try:
        full_sensor_data, time_vector = read_window(
            filePathVibe, start_sec=startTime, end_sec=endTime
        )
    except Exception as e:
        print(f"Error reading data file: {e}")
        return

    print(f"Data loaded. Applying {cutoff_freq} Hz high-pass filter and generating spectrograms...")

    for sensor_name in full_sensor_data.columns:
        raw_signal = full_sensor_data[sensor_name].values
        detrended_signal = detrend(raw_signal)

        # <-- NEW: Apply the high-pass filter after detrending
        filtered_signal = apply_highpass_filter(detrended_signal, cutoff=cutoff_freq, fs=fs)

        # Generate the Spectrogram using the *filtered* signal
        f, t, Sxx = spectrogram(
            filtered_signal,
            fs=fs,
            nperseg=2048,
            noverlap=1024
        )

        plt.figure(figsize=(12, 8))
        Sxx_db = 10 * np.log10(Sxx)
        plt.pcolormesh(t, f, Sxx_db, shading='gouraud', cmap='viridis')

        plt.colorbar(label='Intensity (dB)')
        plt.title(f'Spectrogram for Sensor: {sensor_name} (Filtered > {cutoff_freq} Hz)', fontsize=16)
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.ylim(0, 2500)
        


# --- Main function for Welch-based Spectrogram ---
def create_welch_spectrograms(filePathVibe, startTime, timeWindow, fs=5000, cutoff_freq=100):
    """
    Creates a spectrogram by manually applying Welch's method to sliding windows.
    """
    print(f"Loading {timeWindow} seconds of data starting from {startTime}s...")
    endTime = startTime + timeWindow
    full_sensor_data, _ = read_window(filePathVibe, start_sec=startTime, end_sec=endTime)

    print("Data loaded. Generating Welch-based spectrogram for each sensor...")

    # --- Spectrogram Parameters ---
    window_duration_sec = 0.5  # Duration of each time-slice (e.g., 0.5 seconds)
    step_duration_sec = 0.05   # How much to slide the window forward (e.g., 0.05 seconds)
    
    # Convert seconds to number of samples
    window_samples = int(window_duration_sec * fs)
    step_samples = int(step_duration_sec * fs)

    for sensor_name in full_sensor_data.columns:
        # --- Signal Preparation ---
        raw_signal = full_sensor_data[sensor_name].values
        detrended_signal = detrend(raw_signal)
        filtered_signal = apply_highpass_filter(detrended_signal, cutoff=cutoff_freq, fs=fs)

        # --- Manual Spectrogram Loop ---
        spectrogram_data = []
        time_steps = []
        
        start_index = 0
        while start_index + window_samples <= len(filtered_signal):
            # Extract a chunk of the signal
            signal_chunk = filtered_signal[start_index : start_index + window_samples]
            
            # Run Welch's method on this small chunk
            # nperseg is the length of FFT segments *within* the Welch calculation
            f, Pxx = welch(signal_chunk, fs=fs, nperseg=1024)
            
            spectrogram_data.append(Pxx)
            # The time for this slice is the center of the window
            time_steps.append((start_index + window_samples / 2) / fs)
            
            # Slide the window forward
            start_index += step_samples

        # Convert collected data into a 2D array
        Sxx = np.array(spectrogram_data).T  # Transpose so frequency is on y-axis

        # --- Plotting ---
        plt.figure(figsize=(12, 8))
        Sxx_db = 10 * np.log10(Sxx)
        
        plt.pcolormesh(time_steps, f, Sxx_db, shading='gouraud', cmap='viridis')
        
        plt.colorbar(label='Intensity (dB)')
        plt.title(f'Welch Spectrogram for Sensor: {sensor_name}', fontsize=16)
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.ylim(0, 2500)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import detrend
import pywt

# Assume read_window and apply_highpass_filter are defined as in previous examples

def create_wavelet_scalograms(filePathVibe, startTime, timeWindow, fs=5000, cutoff_freq=100):
    """
    Creates a scalogram for each sensor using a Continuous Wavelet Transform (CWT).
    """
    print(f"Loading {timeWindow} seconds of data starting from {startTime}s...")
    endTime = startTime + timeWindow
    full_sensor_data, time_vector = read_window(
        filePathVibe, start_sec=startTime, end_sec=endTime
    )

    print("Data loaded. Generating wavelet scalogram for each sensor...")

    # --- Wavelet Parameters ---
    # The wavelet to use. Complex Morlet ('cmor') is great for time-frequency analysis.
    wavelet_name = 'cmor1.5-1.0'
    # Define the range of scales to analyze. This corresponds to the frequency range.
    # A smaller number gives higher frequencies, a larger number gives lower frequencies.
    scales = np.arange(10, 250)

    for sensor_name in full_sensor_data.columns:
        # --- Signal Preparation ---
        raw_signal = full_sensor_data[sensor_name].values
        detrended_signal = detrend(raw_signal)
        filtered_signal = apply_highpass_filter(detrended_signal, cutoff=cutoff_freq, fs=fs)

        # --- Step 1: Perform the Continuous Wavelet Transform (CWT) ---
        # The PyWavelets library conveniently returns the coefficients and corresponding frequencies.
        coefficients, frequencies = pywt.cwt(
            filtered_signal,
            scales,
            wavelet_name,
            sampling_period=1/fs
        )
        
        # --- Step 2: Plot the Scalogram ---
        plt.figure(figsize=(12, 8))
        
        # The magnitude of the complex coefficients represents the intensity
        plt.pcolormesh(
            time_vector,
            frequencies,
            np.abs(coefficients),
            shading='gouraud',
            cmap='viridis'
        )

        plt.colorbar(label='Magnitude')
        plt.title(f'Wavelet Scalogram for Sensor: {sensor_name}', fontsize=16)
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.ylim(cutoff_freq, 2500) # Show frequencies above our cutoff
        plt.show()
