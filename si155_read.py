from hyperion import Hyperion
from socket import gaierror

import numpy as np
import os
import h5py
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



