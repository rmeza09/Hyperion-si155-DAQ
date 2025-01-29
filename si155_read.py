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
    def __init__(self, interrogator, sampling_rate=1000, duration=None):
        """
        Initialize the continuous data logger.

        :param interrogator: Interrogator object to fetch data
        :param sampling_rate: Desired sampling rate in Hz
        :param duration: Optional duration to run the data collection (in seconds). If None, runs indefinitely.
        """
        self.interrogator = interrogator
        self.sampling_rate = sampling_rate
        self.duration = duration
        self.data_dir = r"C:\Users\rmeza\Desktop\Hyperion Data Acquisition\DATA"
        self.start_time = None

        # Ensure the DATA directory exists
        os.makedirs(self.data_dir, exist_ok=True)

        # Create a new HDF5 file for this session
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.filename = os.path.join(self.data_dir, f"hyperion_stream_{timestamp}.h5")
        self.hdf_file = h5py.File(self.filename, "w")

        # Create datasets for continuous storage with unlimited size
        self.wavelength_dset = self.hdf_file.create_dataset(
            "wavelengths", (0, 5), maxshape=(None, 5), dtype="float64"
        )
        self.intensity_dset = self.hdf_file.create_dataset(
            "intensities", (0, 5), maxshape=(None, 5), dtype="float64"
        )

    def start_logging(self):
        """Start continuously logging data."""
        print(f"Starting data collection at {self.sampling_rate} Hz...")
        self.start_time = time.time()
        sample_count = 0

        try:
            while self.duration is None or (time.time() - self.start_time) < self.duration:
                start_loop = time.time()

                # Collect data from the interrogator
                peak_data, intensity_data = self.interrogator.getData()

                # Resize datasets to accommodate new data
                self.wavelength_dset.resize((sample_count + 1, 5))
                self.intensity_dset.resize((sample_count + 1, 5))

                # Store new data
                self.wavelength_dset[sample_count, :] = peak_data
                self.intensity_dset[sample_count, :] = intensity_data

                sample_count += 1

                # Enforce sampling rate
                elapsed = time.time() - start_loop
                sleep_time = max(0, (1 / self.sampling_rate) - elapsed)
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("Data collection manually stopped.")

        finally:
            self.hdf_file.close()
            print(f"Data saved in {self.filename}")



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


def main( args=None):
    interrogator = Interrogator("10.0.0.55")
    #print(interrogator.interrogator.is_ready)
    #print(interrogator.is_connected)
    #print(interrogator.num_chs)
    peak_data, intensity_data = interrogator.getData()
    save_to_hdf5(peak_data, intensity_data)

    #print("Peak Wavelengths:", peak_data)
    #print("Intensities:", intensity_data)
    #print(interrogator.getData())
    #print(interrogator.signal_each_ch)

    #print(interrogator.total_reading_num)




if __name__ == "__main__":
    main()



