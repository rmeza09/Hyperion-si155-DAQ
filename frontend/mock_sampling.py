import h5py 
import numpy as np
import pandas as pd
import time

def mock_sampling():

    h5_filepath = "./DATA/hyperion_stream_2025-02-28_12-51-06.h5"

    # Open the file
    with h5py.File(h5_filepath, "r") as hdf:
        print("Keys in HDF5 file:", list(hdf.keys()))  # Show available datasets
        
        # Extract wavelength and intensity data
        wavelengths = hdf["wavelengths"][:]  # Assuming dataset is named 'wavelengths'
        intensities = hdf["intensities"][:]  # Assuming dataset is named 'intensities'

        print("Wavelength Data Shape:", wavelengths.shape)  # (num_samples, num_sensors)
        print("Intensity Data Shape:", intensities.shape)

    for row in range(0, wavelengths.shape[0], 100):
        # Create a Pandas DataFrame for each sensor
        WL_snapshot = wavelengths[row, :]
        Int_snapshot = intensities[row, :]
        sensor_df = pd.DataFrame({"Wavelength (nm)": WL_snapshot, "Intensity (dB)": Int_snapshot})
        yield sensor_df
        time.sleep(0.1)  # Simulate a delay between snapshots



#for df in mock_sampling():
#    print(df)  # Should print one row every 100 samples



