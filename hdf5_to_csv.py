import h5py
import pandas as pd
import os
import numpy as np

def convert_hdf5_to_csv(hdf5_filepath):
    """
    Converts an HDF5 file to a CSV file by extracting wavelength and intensity data.

    :param hdf5_filepath: Path to the input HDF5 file
    """
    # Verify that the file exists
    if not os.path.exists(hdf5_filepath):
        print(f"Error: The file '{hdf5_filepath}' does not exist.")
        return

    # Open the HDF5 file
    with h5py.File(hdf5_filepath, "r") as hdf:
        # Check if the necessary datasets exist
        if "wavelengths" not in hdf or "intensities" not in hdf:
            print(f"Error: The file '{hdf5_filepath}' does not contain the required datasets.")
            return

        # Read datasets
        wavelengths = hdf["wavelengths"][:]
        intensities = hdf["intensities"][:]

    # Create a DataFrame with labeled columns
    df = pd.DataFrame(
        data= np.hstack((wavelengths, intensities)),
        columns=[f"FBG_{i+1}_Wavelength" for i in range(wavelengths.shape[1])] +
                [f"FBG_{i+1}_Intensity" for i in range(intensities.shape[1])]
    )

    # Generate the CSV filename based on the HDF5 filename
    csv_filepath = hdf5_filepath.replace(".h5", ".csv")

    # Save to CSV
    df.to_csv(csv_filepath, index=False)
    print(f"Conversion successful! CSV file saved as: {csv_filepath}")

# ----- Run the script for any HDF5 file -----
if __name__ == "__main__":
    # Manually specify the HDF5 file to convert
    hdf5_file = input("Enter the path of the HDF5 file to convert: ").strip()
    convert_hdf5_to_csv(hdf5_file)
