from hyperion import Hyperion
from socket import gaierror

import numpy as np
import pandas as pd
import os
import h5py
import time
from datetime import datetime

class Interrogator():
    def __init__(self, address, timeout: float = 1):
        self.address = address
        self.is_connected = False
        self.signal_each_ch = np.zeros(4) #number of channels
        self.total_reading_num = 0
        self.available_ch = {}
        self.connect()
        self.check_ch_available()

    def connect( self ):
        self.interrogator = Hyperion( self.address )
        try:
            self.is_ready = self.interrogator.is_ready
            self.is_connected = True
            self.num_chs = self.interrogator.channel_count
            print("interrogator connected")
        except OSError:
            self.is_connected = False
            print("Fail to connect with Interrogator!")

    def check_ch_available(self):
        peaks = self.interrogator.peaks
        for idx in range(1, self.num_chs+1):
            numCHSensors = np.size(peaks[idx].astype( np.float64 ))
            self.available_ch['CH1'] = numCHSensors
            self.signal_each_ch[idx-1] = numCHSensors
            self.total_reading_num += numCHSensors

    def getData(self) -> tuple:
        peaks = self.interrogator.peaks  # Get detected peak wavelengths
        spectra = self.interrogator.spectra  # Get full spectral intensity data

        peak_data = []
        intensity_data = []

        for idx in range(1, self.num_chs+1):
            peak_data = np.concatenate((peak_data, peaks[idx].astype(np.float64)))

            if idx == 3:  # Only process Channel 3
                full_intensities = self.interrogator.spectra[3]  # Full intensity spectrum for CH3
                num_points = len(full_intensities)

                # Compute step size based on known spectral range (1510 to 1590 nm)
                step_size = (1590 - 1510) / (num_points - 1)

                # Find intensity values corresponding to peak wavelengths
                intensity_data = np.concatenate((
                    intensity_data,
                    np.array([
                        full_intensities[int((peak - 1510) / step_size)] for peak in peaks[3]
                    ])
                ))
            else:
                intensity_data = np.concatenate((intensity_data, np.zeros_like(peaks[idx])))

        return peak_data, intensity_data


class ContinuousDataLogger:
    def __init__(self, interrogator, sampling_rate, duration=None):
        """
        Initialize the continuous data logger.

        :param interrogator: Interrogator object to fetch data
        :param sampling_rate: Desired sampling rate in Hz
        :param duration: Optional duration to run the data collection (in seconds). If None, runs indefinitely.
        """
        self.interrogator = interrogator
        self.sampling_rate = sampling_rate
        self.duration = duration
        self.running = False # control flag for knowing if the system is running
        self.data_dir = r"C:\Users\rmeza\Desktop\Hyperion Data Acquisition\DATA"
        self.start_time = datetime.now()

        # Ensure the DATA directory exists
        os.makedirs(self.data_dir, exist_ok=True)

        # Create a new HDF5 file for this session
        timestamp = self.start_time.strftime("%Y-%m-%d_%H-%M-%S")
        self.filename = os.path.join(self.data_dir, f"hyperion_stream_{timestamp}.h5")
        self.hdf_file = h5py.File(self.filename, "w")

        # ---- Add Initial Metadata (Start Time, Sampling Rate) ----
        self.hdf_file.attrs["Device"] = "Hyperion SI155"
        self.hdf_file.attrs["Start Time"] = self.start_time.strftime("%Y-%m-%d %H:%M:%S")
        self.hdf_file.attrs["Sampling Rate (Hz)"] = self.sampling_rate

        # Create datasets for continuous storage with unlimited size
        self.wavelength_dset = self.hdf_file.create_dataset(
            "wavelengths", (0, 5), maxshape=(None, 5), dtype="float64"
        )
        self.intensity_dset = self.hdf_file.create_dataset(
            "intensities", (0, 5), maxshape=(None, 5), dtype="float64"
        )

    def start_logging(self):
        """Start continuously logging data with automatic reconnection."""
        print(f"Collecting data at {self.sampling_rate} Hz...")
        sample_count = 0
        self.running = True  # Set flag to True when logging starts

        try:
            while self.running and (self.duration is None or (time.time() - self.start_time.timestamp()) < self.duration):
                start_loop = time.time()

                # Attempt data collection with auto-reconnect
                while self.running:  # Ensure we break if stopped
                    try:
                        peak_data, intensity_data = self.interrogator.getData()
                        break  # Successful connection, exit loop
                    except Exception as e:
                        print(f"⚠️ Connection lost: {e}. Attempting to reconnect...")

                        # Keep retrying until it reconnects or is stopped
                        while self.running:
                            try:
                                self.interrogator = Interrogator("10.0.0.55")  # Reinitialize connection
                                if self.interrogator.is_connected:
                                    print("✅ Reconnected successfully!")
                                    break  # Exit reconnect loop
                            except Exception as reconnect_error:
                                print(f"🔄 Retrying connection...")

                if not self.running:
                    break  # Stop logging if flag is False

                # Resize datasets for new data
                self.wavelength_dset.resize((sample_count + 1, 5))
                self.intensity_dset.resize((sample_count + 1, 5))

                # Store new data
                self.wavelength_dset[sample_count, :] = peak_data
                self.intensity_dset[sample_count, :] = intensity_data

                sample_count += 1

                # Maintain sampling rate
                elapsed = time.time() - start_loop
                sleep_time = max(0, (1 / self.sampling_rate) - elapsed)
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("🛑 Data collection manually stopped.")

        finally:
            self.stop_logging()  # Ensure cleanup when stopped
            
    def stop_logging(self):
        """Stops the continuous data logging gracefully."""
        print("🛑 Stopping data logging...")
        self.running = False  # Set control flag to False

        # Ensure the HDF5 file is properly closed
        if self.hdf_file:
            end_time = datetime.now()
            total_duration = (end_time - self.start_time).total_seconds()

            # Update metadata in HDF5 file
            with h5py.File(self.filename, "a") as hdf_file:
                hdf_file.attrs["End Time"] = end_time.strftime("%Y-%m-%d %H:%M:%S")
                hdf_file.attrs["Total Duration (s)"] = total_duration

            self.hdf_file.close()
            print(f"📁 Data saved in {self.filename}")
            print(f"Test ended at {end_time.strftime('%Y-%m-%d %H:%M:%S')}, Total duration: {total_duration:.2f} seconds")
    
    def get_latest_snapshot(self):
        """Retrieve the most recent snapshot of wavelength data for UI updates."""
        try:
            with h5py.File(self.filename, "r") as hdf:
                if "wavelengths" in hdf:
                    last_index = hdf["wavelengths"].shape[0] - 1  # Last recorded row
                    if last_index >= 0:
                        latest_wavelengths = hdf["wavelengths"][last_index, :]
                        return pd.DataFrame({"Wavelength (nm)": latest_wavelengths})
        except Exception as e:
            print(f"⚠️ Error retrieving snapshot: {e}")
        return None  # Return None if no valid data is found


    def save_to_hdf5(peak_data, intensity_data):
        # Ensure the DATA directory exists
        data_dir = "DATA"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        # Generate a unique filename with the current date and time
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = os.path.join(data_dir, f"hyperion_data_{timestamp}.h5")

        # Create HDF5 file and store datasets
        with h5py.File(filename, "w") as hdf_file:
            hdf_file.create_dataset("wavelengths", data=peak_data, dtype="float64")
            hdf_file.create_dataset("intensities", data=intensity_data, dtype="float64")

        print(f"Data successfully saved to {filename}")

'''
def main(args=None):
    interrogator = Interrogator("10.0.0.55")

    # Set desired sampling rate & duration
    sampling_rate = 5000  # Hz
    duration = None  # 600 seconds (10 minutes) OR set to None for infinite logging

    # Start continuous data logging
    logger = ContinuousDataLogger(interrogator, sampling_rate=sampling_rate, duration=duration)
    logger.start_logging()

if __name__ == "__main__":
    main()
'''



