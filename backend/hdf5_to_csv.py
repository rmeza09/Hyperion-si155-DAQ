import h5py
import pandas as pd
import os
import numpy as np

def convert_hdf5_to_csv(hdf5_filepath, output_folder="DATA/CSV"):
    """
    Converts an HDF5 file to a CSV file by extracting wavelength and intensity data.

    :param hdf5_filepath: Path to the input HDF5 file
    :param output_folder: Folder to store the output CSV file
    """
    # Verify that the file exists
    if not os.path.exists(hdf5_filepath):
        print(f"Error: The file '{hdf5_filepath}' does not exist.")
        return

    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Open the HDF5 file
    with h5py.File(hdf5_filepath, "r") as hdf:
        # List available datasets
        datasets = list(hdf.keys())
        print(f"Datasets available in {hdf5_filepath}: {datasets}")

        # Validate required datasets exist
        if "wavelengths" not in datasets or "intensities" not in datasets:
            print(f"Error: Required datasets ('wavelengths', 'intensities') not found in {hdf5_filepath}.")
            return

        # Read datasets
        wavelengths = hdf["wavelengths"][:]
        intensities = hdf["intensities"][:]

    # Create DataFrame with labeled columns
    df = pd.DataFrame(
        data=np.hstack((wavelengths, intensities)),
        columns=[f"FBG_{i+1}_Wavelength" for i in range(wavelengths.shape[1])] +
                [f"FBG_{i+1}_Intensity" for i in range(intensities.shape[1])]
    )

    # Generate a CSV filename in the output folder
    csv_filename = os.path.join(output_folder, os.path.basename(hdf5_filepath).replace(".h5", ".csv"))

    # Save to CSV
    df.to_csv(csv_filename, index=False)
    print(f"✅ Conversion successful! CSV file saved as: {csv_filename}")

