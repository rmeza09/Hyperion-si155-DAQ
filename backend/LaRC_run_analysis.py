import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, windows, welch, detrend
from matplotlib.ticker import ScalarFormatter

from FFT_Analysis_Functions import fileReader, read_window, signal_detrend_analysis


# filePathVibe = './DATA/LaRC_Data/LaRC_test1_run0008.txt' #set startTime = 180 timewWindow = 20
# filePathVibe = './DATA/LaRC_Data/LaRC_test2_run0010.txt' #set startTime = 210 timewWindow = 40
# filePathVibe = './DATA/LaRC_Data/LaRC_test3_run0012_bad.txt' # not usable
# filePathVibe = './DATA/LaRC_Data/LaRC_test4_run0022_noKuliteData.txt' # not usable
# filePathVibe = './DATA/LaRC_Data/LaRC_test5_run0024.txt' #set startTime = 95 timewWindow = 15
# filePathVibe = './DATA/LaRC_Data/LaRC_test6_run0026.txt' #set startTime = 175 timewWindow = 15
# filePathVibe = './DATA/LaRC_Data/LaRC_test7_run0028.txt' #set startTime = 80 timewWindow = 15
# filePathVibe = './DATA/LaRC_Data/LaRC_test8_run0030.txt' #set startTime = 180 timewWindow = 25
# filePathVibe = './DATA/LaRC_Data/LaRC_test9_run0032.txt' #set startTime = 55 timewWindow = 15
# filePathVibe = './DATA/LaRC_Data/LaRC_test10_run0034.txt' #set startTime = 80 timewWindow = 25
# filePathVibe = './DATA/LaRC_Data/LaRC_test11_run0036.txt' #set startTime =  timewWindow = 
# filePathVibe = './DATA/LaRC_Data/LaRC_test12_run0038.txt' #set startTime =  timewWindow = 
# filePathVibe = './DATA/LaRC_Data/LaRC_test13_run0040.txt' #set startTime =  timewWindow = 
# filePathVibe = './DATA/LaRC_Data/LaRC_test14_run0042.txt' #set startTime =  timewWindow = 
# filePathVibe = './DATA/LaRC_Data/LaRC_test15_run0044_5secIRdata.txt' #set startTime =  timewWindow = 
# filePathVibe = './DATA/LaRC_Data/LaRC_test16_run046_noKuliteData.txt' #set startTime =  timewWindow = 
# filePathVibe = './DATA/LaRC_Data/LaRC_test17_run0048.txt' #set startTime =  timewWindow = 
filePathVibe = './DATA/LaRC_Data/LaRC_test18_run0050.txt' #set startTime = 80 timewWindow = 20
# filePathVibe = './DATA/LaRC_Data/LaRC_test19_run0052_noIRdata.txt' #set startTime =  timewWindow = 


startTime = 80
timeWindow = 20
cutoff = 100
dfVibe, timeSpanV, samplingFreqV = fileReader(filePathVibe, False, True, True, start_trim=startTime) # filePath, flip, plot, indivPlot, start_trim=0
signal_detrend_analysis(filePathVibe, startTime, timeWindow, cutoff, plotSegments=True, plotFFT=False, saveFreqs=True, saveFigures=False) # filePathVibe, startTime, timeWindow, cutoff, plotSegments=False, plotFFT=False, saveFreqs=False, saveFigures=False