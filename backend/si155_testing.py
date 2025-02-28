from hyperion import Hyperion
from socket import gaierror

import numpy as np
import os
import h5py
import time
from datetime import datetime

class Interrogator():
    def __init__(self, address, timeout: float = 1):
        self.address = address
        self.is_connected = False
        self.signal_each_ch = np.zeros(4)  # Number of channels
        self.total_reading_num = 0
        self.available_ch = {}
        self.connect()
        self.check_ch_available()

    def connect(self):
        try:
            self.interrogator = Hyperion(self.address)
            self.is_ready = self.interrogator.is_ready
            self.is_connected = True
            self.num_chs = self.interrogator.channel_count
            print("✅ Interrogator connected")
        except OSError:
            self.is_connected = False
            print("❌ Failed to connect with Interrogator!")

    def check_ch_available(self):
        peaks = self.interrogator.peaks
        for idx in range(1, self.num_chs + 1):
            numCHSensors = np.size(peaks[idx].astype(np.float64))
            self.available_ch[f'CH{idx}'] = numCHSensors
            self.signal_each_ch[idx - 1] = numCHSensors
            self.total_reading_num += numCHSensors

    def getData(self) -> tuple:
        try:
            peaks = self.interrogator.peaks  # Get detected peak wavelengths
            spectra = self.interrogator.spectra  # Get full spectral intensity data

            peak_data, intensity_data = [], []

            for idx in range(1, self.num_chs + 1):
                peak_data = np.concatenate((peak_data, peaks[idx].astype(np.float64)))

                if idx == 3:  # Only process Channel 3
                    full_intensities = self.interrogator.spectra[3]  # Full intensity spectrum for CH3
                    num_points = len(full_intensities)
                    step_size = (1590 - 1510) / (num_points - 1)

                    intensity_data = np.concatenate((
                        intensity_data,
                        np.array([
                            full_intensities[int((peak - 1510) / step_size)] for peak in peaks[3]
                        ])
                    ))
                else:
                    intensity_data = np.concatenate((intensity_data, np.zeros_like(peaks[idx])))

            print(f"✅ Data retrieved: {len(peak_data)} wavelengths, {len(intensity_data)} intensities")
            return peak_data, intensity_data
        except Exception as e:
            print(f"⚠️ Error retrieving data: {e}")
            return [], []

class ContinuousDataLogger:
    def __init__(self, interrogator, sampling_rate=1000, duration=None):
        self.interrogator = interrogator
        self.sampling_rate = sampling_rate
        self.duration = duration
        self.data_dir = r"C:\\Users\\rmeza\\Desktop\\Hyperion Data Acquisition\\DATA"
        self.start_time = datetime.now()

        os.makedirs(self.data_dir, exist_ok=True)
        timestamp = self.start_time.strftime("%Y-%m-%d_%H-%M-%S")
        self.filename = os.path.join(self.data_dir, f"hyperion_stream_{timestamp}.h5")
        self.hdf_file = h5py.File(self.filename, "w")

        self.hdf_file.attrs["Device"] = "Hyperion SI155"
        self.hdf_file.attrs["Start Time"] = self.start_time.strftime("%Y-%m-%d %H:%M:%S")
        self.hdf_file.attrs["Sampling Rate (Hz)"] = self.sampling_rate

        self.wavelength_dset = self.hdf_file.create_dataset("wavelengths", (0, 5), maxshape=(None, 5), dtype="float64")
        self.intensity_dset = self.hdf_file.create_dataset("intensities", (0, 5), maxshape=(None, 5), dtype="float64")

    def start_logging(self):
        print(f"🔴 Collecting data at {self.sampling_rate} Hz...")
        sample_count = 0
        max_samples = 5  # Hardcoded limit to collect exactly 5 samples

        try:
            while sample_count < max_samples:
                print(f"Loop iteration {sample_count}")
                peak_data, intensity_data = self.interrogator.getData()

                if len(peak_data) > 0 and len(intensity_data) > 0:
                    self.wavelength_dset.resize((sample_count + 1, 5))
                    self.intensity_dset.resize((sample_count + 1, 5))
                    self.wavelength_dset[sample_count, :] = peak_data
                    self.intensity_dset[sample_count, :] = intensity_data
                    sample_count += 1
                    self.hdf_file.flush()
                    print(f"✅ Sample {sample_count} collected")
                else:
                    print("⚠️ No valid data received; retrying...")

        except KeyboardInterrupt:
            print("🛑 Data collection manually stopped.")

        finally:
            self.hdf_file.attrs["End Time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.hdf_file.attrs["Total Duration (s)"] = (datetime.now() - self.start_time).total_seconds()
            self.hdf_file.close()
            print(f"📁 Data saved in {self.filename}")

if __name__ == "__main__":
    interrogator = Interrogator("10.0.0.55")
    logger = ContinuousDataLogger(interrogator, sampling_rate=1000, duration=600)
    logger.start_logging()
