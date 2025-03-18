from hyperion import Hyperion
from socket import gaierror

import numpy as np
import os
import h5py
import time
from time import perf_counter
from datetime import datetime

import asyncio
from hyperion import HCommTCPPeaksStreamer

class Interrogator():
    def __init__(self, address, timeout: float = 1):
        self.address = address
        self.is_connected = False
        self.queue = asyncio.Queue()  # Streaming queue for incoming data
        self.streamer = None  # Peak streamer instance
        asyncio.create_task(self.connect())

    async def connect(self):
        """Initialize connection with Hyperion and start streaming."""
        try:
            loop = asyncio.get_running_loop()  # Ensure we use an active event loop
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        try:  # Move the try block down to properly catch connection errors
            self.streamer = HCommTCPPeaksStreamer(self.address, loop=loop, queue=self.queue)
            self.is_connected = True
            print("✅ Interrogator connected and streaming started.")

            # Start streaming properly in background
            asyncio.create_task(self.streamer.stream_data())  # 🚀 Correct way to run in an async function

        except Exception as e:
            self.is_connected = False
            print(f"⚠️ Connection failed: {e}")  # ✅ Now correctly indented



    async def getData(self) -> np.ndarray:
        """Asynchronously retrieve peak data from the stream with timeout to prevent blocking."""
        start_time = time.perf_counter_ns()  # High-precision timer

        try:
            if self.queue.empty():
                print("⚠️ No data in queue, returning empty array.")
                return np.array([])

            peak_data = await asyncio.wait_for(self.queue.get(), timeout=1.0)  # 🚀 Timeout if no data in 1 second
            print(f"📡 Retrieved data from queue: {peak_data}")  # Debugging: Show received data

            peak_data = peak_data[3].astype(np.float64)  # Extract Channel 3 and convert

        except asyncio.TimeoutError:
            print("⚠️ Timeout waiting for data from queue. Returning empty array.")
            return np.array([])

        except Exception as e:
            print(f"⚠️ Error retrieving streamed data: {e}")
            return np.array([])

        elapsed = (time.perf_counter_ns() - start_time) / 1e9  # Convert to seconds
        print(f"getData() execution time: {elapsed:.9f} sec")  # Log execution time

        return peak_data



'''
    def getData(self) -> np.ndarray:
        start_time = time.perf_counter_ns()  # More precise timing

        peaks = self.interrogator.peaks  # Retrieve detected peak wavelengths

        # Use NumPy hstack for efficient array merging instead of slow loops
        peak_data = np.hstack([peaks[idx].astype(np.float64) for idx in range(1, self.num_chs+1)])

        elapsed = (time.perf_counter_ns() - start_time) / 1e9  # Convert nanoseconds to seconds
        print(f"getData() execution time: {elapsed:.9f} sec")  # Higher precision logging

        return peak_data

    def getData(self) -> tuple:
        start_time = time.time()  # Track start time
        peaks = self.interrogator.peaks  # Get detected peak wavelengths
        #spectra = self.interrogator.spectra  # Get full spectral intensity data
        elapsed = time.time() - start_time  # Compute execution time
        print(f"getData() execution time: {elapsed:.6f} sec") 

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
'''
    



class ContinuousDataLogger:
    
    def __init__(self, interrogator, sampling_rate=100000, duration=None):
        """
        Initialize the continuous data logger.

        :param interrogator: Interrogator object to fetch data
        :param sampling_rate: Desired sampling rate in Hz
        :param duration: Optional duration to run the data collection (in seconds). If None, runs indefinitely.
        """
        self.interrogator = interrogator
        self.sampling_rate = sampling_rate
        self.duration = duration
        self.running = False  # ✅ Used for stopping logging gracefully
        self.data_dir = r"C:/Users/rmeza/Desktop/Hyperion Data Acquisition/DATA"
        self.start_time = datetime.now()

        # Ensure the DATA directory exists
        os.makedirs(self.data_dir, exist_ok=True)

        # Create or open an HDF5 file for this session
        timestamp = self.start_time.strftime("%Y-%m-%d_%H-%M-%S")
        self.filename = os.path.join(self.data_dir, f"hyperion_stream_{timestamp}.h5")
        self.hdf_file = h5py.File(self.filename, "a")  # ✅ Append mode instead of overwrite

        # ---- Add Initial Metadata (Start Time, Sampling Rate) ----
        self.hdf_file.attrs["Device"] = "Hyperion SI155"
        self.hdf_file.attrs["Start Time"] = self.start_time.strftime("%Y-%m-%d %H:%M:%S")
        self.hdf_file.attrs["Sampling Rate (Hz)"] = self.sampling_rate

        # Create dataset for wavelengths (peak data)
        self.wavelength_dset = self.hdf_file.create_dataset(
            "wavelengths", (0, 5), maxshape=(None, 5), dtype="float64"
        )

        # ✅ Remove `intensity_dset` if not needed (saves space & speed)
        print(f"📁 HDF5 File Initialized: {self.filename}")




    async def start_logging(self):
        """Start continuously logging data asynchronously with batch HDF5 writing."""
        print(f"Collecting data at {self.sampling_rate} Hz...")
        sample_count = 0
        self.running = True  # ✅ Restore control flag for graceful stopping

        batch_size = 100  # 🚀 Collect 100 samples before writing to HDF5
        peak_buffer = []

        try:
            while self.running and (self.duration is None or (time.perf_counter() - self.start_time.timestamp()) < self.duration):
                loop_start = time.perf_counter()
                print('hello')
                # Attempt data collection with auto-reconnect
                try:
                    peak_data = await self.interrogator.getData()  # ✅ Retrieve streaming data
                    print(f"Retrieved data: {peak_data}")  # Debug: See if data is being collected
                    peak_buffer.append(peak_data)

                except Exception as e:
                    print(f"⚠️ Connection lost: {e}. Attempting to reconnect...")

                    # Keep retrying until it reconnects
                    while self.running:
                        try:
                            self.interrogator = Interrogator("10.0.0.55")  # Reinitialize connection
                            if self.interrogator.is_connected:
                                print("✅ Reconnected successfully!")
                                break  # Exit reconnect loop
                        except Exception as reconnect_error:
                            print(f"🔄 Retrying connection...")

                # ✅ Ensure logging stops gracefully
                if not self.running:
                    break  

                sample_count += 1  # ✅ Increment after successful data collection

                # 🚀 Batch writing to HDF5 every `batch_size` samples
                if sample_count % batch_size == 0 and len(peak_buffer) > 0:
                    self.wavelength_dset.resize((sample_count, 5))  # ✅ Resize correctly
                    self.wavelength_dset[sample_count - batch_size:sample_count, :] = np.array(peak_buffer)  # ✅ Append new data
                    peak_buffer = []  # ✅ Clear buffer after writing

                # ✅ Maintain sampling rate with high-precision timing
                while (time.perf_counter() - loop_start) < (1 / self.sampling_rate):
                    await asyncio.sleep(0)

        except KeyboardInterrupt:
            print("🛑 Data collection manually stopped.")

        finally:
            end_time = datetime.now()
            total_duration = (end_time - self.start_time).total_seconds()

            # ✅ Final HDF5 Write Before Closing
            if len(peak_buffer) > 0:
                self.wavelength_dset.resize((sample_count, 5))  # ✅ Resize properly
                self.wavelength_dset[sample_count - len(peak_buffer):sample_count, :] = np.array(peak_buffer)  # ✅ Append remaining data

            # ✅ Ensure the HDF5 file closes properly
            self.hdf_file.close()

            # ✅ Update metadata
            with h5py.File(self.filename, "a") as hdf_file:
                hdf_file.attrs["End Time"] = np.bytes_(end_time.strftime("%Y-%m-%d %H:%M:%S"))
                hdf_file.attrs["Total Duration (s)"] = total_duration

            print(f"📁 Data saved in {self.filename}")
            print(f"Test ended at {end_time.strftime('%Y-%m-%d %H:%M:%S')}, Total duration: {total_duration:.2f} seconds")

    def save_to_hdf5(peak_data):#, intensity_data):
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
            #hdf_file.create_dataset("intensities", data=intensity_data, dtype="float64")

        print(f"Data successfully saved to {filename}")


async def main():
    interrogator = Interrogator("10.0.0.55")
    await interrogator.connect()  # ✅ Await the async connection method

    # Set desired sampling rate & duration
    sampling_rate = 5000  # Hz
    duration = None  # 600 seconds (10 minutes) OR set to None for infinite logging

    # Start continuous data logging
    logger = ContinuousDataLogger(interrogator, sampling_rate=sampling_rate, duration=duration)

    try:
        await logger.start_logging()  # ✅ Ensure this is awaited properly
    except KeyboardInterrupt:
        print("🛑 Data collection manually stopped.")
    finally:
        print("🔄 Shutting down logger and closing files...")
        logger.hdf_file.close()
        print("✅ Cleanup complete.")

# ✅ Explicitly start the event loop
if __name__ == "__main__":
    asyncio.run(main())






