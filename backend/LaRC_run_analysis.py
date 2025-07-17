import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, windows, welch, detrend
from matplotlib.ticker import ScalarFormatter

from FFT_Analysis_Functions import fileReader, read_window, signal_detrend_analysis


#filePathVibe = './DATA/LaRC_Data/LaRC_test1_run0008.txt' #set startTime = 180 timewWindow = 20
filePathVibe = './DATA/LaRC_Data/LaRC_test2_run0010.txt'
# filePathVibe = './DATA/LaRC_Data/LaRC_test3_run0012_bad.txt'
# filePathVibe = './DATA/LaRC_Data/LaRC_test4_run0022_noKuliteData.txt'
# filePathVibe = './DATA/LaRC_Data/LaRC_test5_run0024.txt'
# filePathVibe = './DATA/LaRC_Data/LaRC_test6_run0026.txt'
# filePathVibe = './DATA/LaRC_Data/LaRC_test7_run0028.txt'
#filePathVibe = './DATA/LaRC_Data/LaRC_test8_run0030.txt'
# filePathVibe = './DATA/LaRC_Data/LaRC_test9_run0032.txt'
# filePathVibe = './DATA/LaRC_Data/LaRC_test10_run0034.txt'
# filePathVibe = './DATA/LaRC_Data/LaRC_test11_run0036.txt'
# filePathVibe = './DATA/LaRC_Data/LaRC_test12_run0038.txt'
# filePathVibe = './DATA/LaRC_Data/LaRC_test13_run0040.txt'
# filePathVibe = './DATA/LaRC_Data/LaRC_test14_run0042.txt'
# filePathVibe = './DATA/LaRC_Data/LaRC_test15_run0044_5secIRdata.txt'
# filePathVibe = './DATA/LaRC_Data/LaRC_test16_run046_noKuliteData.txt'
# filePathVibe = './DATA/LaRC_Data/LaRC_test17_run0048.txt'
# filePathVibe = './DATA/LaRC_Data/LaRC_test18_run0050.txt'
# filePathVibe = './DATA/LaRC_Data/LaRC_test19_run0052_noIRdata.txt'

#filePathVibe = './DATA/ENLIGHT/LaRC_Data/LaRC_test8_run0030.txt'
#filePathTemp = './DATA/ENLIGHT/Peaks.20250325154942_TempTest.txt'




startTime = 210
timeWindow = 40
cutoff = 100
dfVibe, timeSpanV, samplingFreqV = fileReader(filePathVibe, False, True, start_trim=startTime)
signal_detrend_analysis(filePathVibe, startTime, timeWindow, cutoff, plotSegments=False, plotFFT=True, saveFreqs=True, saveFigures=False)