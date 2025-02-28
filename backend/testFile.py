import h5py


file_to_read = r"C:\Users\rmeza\Desktop\Hyperion Data Acquisition\DATA\hyperion_data_2025-01-29_18-37-09.h5"

with h5py.File(file_to_read, "r") as hdf:
    print("Stored Datasets:", list(hdf.keys()))
    print("Wavelength Data:", hdf["wavelengths"][:])
    print("Intensity Data:", hdf["intensities"][:])

