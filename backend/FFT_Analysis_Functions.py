import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, windows, welch, detrend, butter, filtfilt
from matplotlib.ticker import ScalarFormatter
import os


def fileReader(filePath, flip, plot, start_trim=0): 
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


def signal_detrend_analysis(filePathVibe, startTime, timeWindow, cutoff, plotSegments=False, plotFFT=False, saveFigures=False):
    # Save location: your system's Downloads folder (cross-platform)
    downloads_path = os.path.join(os.path.expanduser("~"), "Downloads")
    
    # Store all peak frequencies here (1 row = 1 sec window; each row has 5x6 = 30 values)
    peak_matrix = []

    for i in range(0, timeWindow):
        currentTime = startTime + i   
        sensorWL_window, time_window = read_window(filePathVibe, start_sec=currentTime, end_sec=currentTime+1)

        print(sensorWL_window)
        print(time_window)

        if plotSegments:
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

        if plotFFT:
            fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(12, 10))
            ax = ax.flatten()
            fig.suptitle("Frequency Analysis per sensor output", fontsize=16)

        fs = 5000
        cutoff_freq = cutoff
        peak_row = []

        for j, col in enumerate(sensorWL_window.columns):
            detrended_signal = detrend(sensorWL_window[col])
            filtered_signal = apply_highpass_filter(detrended_signal, cutoff_freq, fs, order=4)
            positive_freqs, positive_power = welch(filtered_signal, fs=fs, nperseg=1024, noverlap=512)
            
            # Prominence is dynamic based on signal power scale
            min_prom = 0.15 * np.max(positive_power)
            peak_freqs, _ = find_prominent_peaks(
                positive_freqs, positive_power,
                min_freq=0, max_freq=2500,
                min_prominence=min_prom,
                top_n=6
            )

            # Pad to 6 values if fewer peaks are found
            padded = np.pad(peak_freqs, (0, 6 - len(peak_freqs)), constant_values=np.nan)
            peak_row.extend(padded)

            if plotFFT:
                ax[j].plot(positive_freqs, positive_power)
                ax[j].plot(peak_freqs, [_ for _ in _], 'ro', label='Peaks')
                ax[j].set_title(col)
                ax[j].set_xlabel("Frequency (Hz)")
                ax[j].set_ylabel("Power")

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
            

    # Create DataFrame and save to Excel
    sensor_labels = list(sensorWL_window.columns)
    col_names = [f"{label}_Peak{i+1}" for label in sensor_labels for i in range(6)]
    peak_df = pd.DataFrame(peak_matrix, columns=col_names)
    excel_filename = f"PeakFrequencies_{startTime}s_to_{startTime + timeWindow}s.xlsx"
    excel_path = os.path.join(downloads_path, excel_filename)
    peak_df.to_excel(excel_path, index=False)
    print(f"Peak frequencies saved to: {excel_path}")

    plt.show()
    plt.close()
