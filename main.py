from frontend.main_UI import HyperionDAQUI  # Import the UI class
from backend.si155_read import Interrogator, ContinuousDataLogger # Import the interrogator class

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QThread
import sys

class InterrogatorThread(QThread):
    """Thread to initialize the interrogator to prevent UI blocking."""
    def __init__(self, ip_address):
        super().__init__()
        self.interrogator = None
        self.ip_address = ip_address
        self.num_sensors = 0  # Default to 0 until updated

    def run(self):
        self.interrogator = Interrogator(self.ip_address)
        peak_data, intensity_data = self.interrogator.getData()
        self.num_sensors = len(peak_data)

def main():
    
    # Start Interrogator in separate thread (prevents UI freeze)
    interrogator_thread = InterrogatorThread("10.0.0.55")
    interrogator_thread.start()
    interrogator_thread.wait() 

    # Set desired sampling rate & duration
    sampling_rate = 5000  # Hz
    duration = None  # 600 seconds (10 minutes) OR set to None for infinite logging

    app = QApplication(sys.argv)  # Initialize PyQt
    window = HyperionDAQUI(interrogator_thread.num_sensors)  # Create the UI instance
    data_logger = ContinuousDataLogger(interrogator_thread, sampling_rate=sampling_rate, duration=duration)

    # Connect UI buttons to backend functions
    window.start_button.clicked.connect(data_logger.start_logging)
    window.stop_button.clicked.connect(data_logger.stop_logging)

    window.show()  # Show the UI
    sys.exit(app.exec())  # Run the app

if __name__ == "__main__":
    main()
