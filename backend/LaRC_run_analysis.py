import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, windows, welch
from matplotlib.ticker import ScalarFormatter

from FFT_Analysis_Functions import fileReader, read_window, extract_impacts, apply_hamming_correction, single_fft_analysis


#filePathVibe = './DATA/ENLIGHT/LaRC_test10_run0034.txt' 
#filePathVibe = './DATA/ENLIGHT/LaRC_test9_run0032.txt' 
filePathVibe = './DATA/ENLIGHT/LaRC_test8_run0030.txt'
#filePathTemp = './DATA/ENLIGHT/Peaks.20250325154942_TempTest.txt'

dfVibe, timeSpanV, samplingFreqV = fileReader(filePathVibe, False, True, start_trim=125)

sensorWL_window, time_window = read_window(filePathVibe, start_sec=125, end_sec=135)
positive_freqs, positive_power = single_fft_analysis(sensorWL_window, time_window)

plt.figure()
plt.plot(positive_freqs, positive_power)

plt.show()

