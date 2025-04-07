
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, windows, welch
from matplotlib.ticker import ScalarFormatter

from FFT_Analysis_Functions import fileReader, apply_hamming_correction, single_fft_analysis

def analyze_signal(filepath, testFrequency, save):
    dfOutput, timeSpan, samplingFreq = fileReader(filepath, False, False)
    WLspectrum = np.array(dfOutput.iloc[:, 4])
    time =np.array(timeSpan)

    time_target = 10
    closest_index = (np.abs(time - time_target)).argmin()

    # print("Closest index to 10 seconds:", closest_index)

    WLspectrum = WLspectrum[:closest_index]
    time = time[:closest_index]

    # print(dfWavelength.head())
    # print(WLspectrum.shape)
    # print(time.shape)

    correctedSignal = apply_hamming_correction(WLspectrum, time)

    freq, power = single_fft_analysis(correctedSignal, time)

    peak_index = np.argmax(power)  # Index of the maximum power
    peak_freq = freq[peak_index]  # Frequency at the peak
    peak_power = power[peak_index]  # Maximum power value

    fig, axes = plt.subplots(3, 1, figsize=(10, 8))

    # --- Plot 1: Raw Signal ---
    axes[0].plot(time, WLspectrum)
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Wavelength (nm)')
    axes[0].set_title('Wavelength Change vs Time @ ' + str(testFrequency) + 'Hz')
    axes[0].grid(True)
    
    # --- Plot 2: Hamming-Corrected Signal ---
    axes[1].plot(time, correctedSignal)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Wavelength (nm)')
    axes[1].set_title('Corrected Wavelength Change vs Time @ ' + str(testFrequency) + 'Hz')
    axes[1].grid(True)

    # --- Plot 3: FFT Result ---
    axes[2].plot(freq, power)
    axes[2].scatter(peak_freq, peak_power, color='red', label=f'Peak: {peak_freq:.2f} Hz', zorder=5)  # Red dot at the peak
    axes[2].set_xlabel('Frequency (Hz)')
    axes[2].set_ylabel('Power')
    axes[2].set_title('FFT Power Spectrum')
    axes[2].grid(True)

    # Disable scientific notation on all y-axes
    formatter = ScalarFormatter(useMathText=False)
    formatter.set_scientific(False)
    formatter.set_useOffset(False)

    for ax in axes:
        ax.yaxis.set_major_formatter(formatter)

    plt.tight_layout()

    if save:
        # Save to Downloads
        downloads_path = os.path.expanduser("~/Downloads")
        base_filename = os.path.basename(filepath).replace('.txt', '')
        save_filename = f"{base_filename}_{testFrequency}Hz.png"
        save_path = os.path.join(downloads_path, save_filename)
        plt.savefig(save_path)
        print(f"Saved plot to: {save_path}")


folder_path = './DATA/ENLIGHT/Calibration Shaker Data'
testFrequencies = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000]

file_list = sorted([f for f in os.listdir(folder_path) if f.endswith('.txt')])

for i, filename in enumerate(file_list):
    full_path = os.path.join(folder_path, filename)
    freq_label = testFrequencies[i // 2]  # Every 2 files correspond to the same frequency
    print(f"\nProcessing file: {filename} (Expected Frequency: {freq_label} Hz)")
    analyze_signal(full_path, freq_label, True)



plt.show()