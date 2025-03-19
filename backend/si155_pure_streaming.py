import asyncio
import numpy as np
import time
from hyperion import AsyncHyperion, HCommTCPPeaksStreamer

class AsyncPeakCollector:
    def __init__(self, address):
        self.address = address
        self.loop = asyncio.get_event_loop()
        self.queue = asyncio.Queue()
        self.peak_data = np.empty((0,))  # Initialize empty array for stacking
        self.start_time = None
        self.streamer = None
        self.stop_event = asyncio.Event()

    async def collect_data(self):
        """Collect peak data asynchronously."""
        self.start_time = time.time()
        
        # Initialize async Hyperion instrument connection
        hyperion = AsyncHyperion(self.address, loop=self.loop)
        
        try:
            is_ready = await hyperion.get_is_ready()
            if not is_ready:
                print("Hyperion device is not ready. Exiting.")
                return
        except Exception as e:
            print(f"Failed to communicate with Hyperion: {e}")
            return

        self.streamer = HCommTCPPeaksStreamer(self.address, loop=self.loop, queue=self.queue)

        # Start streaming data
        streaming_task = asyncio.create_task(self.streamer.stream_data())
        print("Started peak data collection...")

        try:
            while not self.stop_event.is_set():
                data_packet = await self.queue.get()
                
                if data_packet["data"] is None:
                    print("Received empty data packet. Stopping stream.")
                    break  # Stop condition for streaming
                
                peaks = data_packet["data"].data.astype(np.float64)
                print(f"Received peak data: {peaks.shape}")  # Debugging log
                
                # Append to existing dataset
                self.peak_data = np.vstack([self.peak_data, peaks]) if self.peak_data.size else peaks
        
        except asyncio.CancelledError:
            print("Streaming cancelled.")
        finally:
            await self.stop_collection()

    async def stop_collection(self):
        """Gracefully stop streaming and print results."""
        print("Stopping data collection...")
        self.stop_event.set()
        if self.streamer:
            self.streamer.stop_streaming()
        
        elapsed_time = time.time() - self.start_time
        samples_collected = self.peak_data.shape[0]
        sampling_rate = samples_collected / elapsed_time if elapsed_time > 0 else 0

        print(f"\nData collection stopped.")
        print(f"Time Elapsed: {elapsed_time:.2f} sec")
        print(f"Total Samples Collected: {samples_collected}")
        print(f"Estimated Sampling Frequency: {sampling_rate:.2f} Hz")

async def main():
    collector = AsyncPeakCollector("10.0.0.55")
    
    task = asyncio.create_task(collector.collect_data())
    
    try:
        await task
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt detected. Stopping...")
        await collector.stop_collection()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    finally:
        print("Shutdown complete.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except RuntimeError:
        # Workaround for event loop closure issue on Windows
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(main())