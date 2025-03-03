from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QTableWidget, QTableWidgetItem, QLabel, QPushButton, QComboBox
from PyQt6.QtCore import QThread, pyqtSignal
import pyqtgraph as pg
import sys
import pandas as pd
import time
import numpy as np
import json

from mock_sampling import mock_sampling  # Import mock data generator

CONFIG_FILE = "sensor_array.json"

class DataUpdateThread(QThread):
    """Separate thread for streaming mock sensor data."""
    data_updated = pyqtSignal(pd.DataFrame)  # Signal to send new data to UI

    def run(self):
        for df in mock_sampling():  # Continuously fetch data from mock function
            self.data_updated.emit(df)  # Emit signal with new data
            time.sleep(0.1)  # Simulate real-time behavior

class HyperionDAQUI(QMainWindow):
    def __init__(self):
        super().__init__()

        # Window Settings
        self.setWindowTitle("Hyperion SI155 DAQ")
        self.setGeometry(500, 100, 1400, 800)  # Adjusted for better layout

        # Hardcoded number of sensors (for now, later will be dynamic)
        self.num_sensors = 5  # This will later be replaced with a backend call
        self.is_active = False

        self.central_wavelengths = np.zeros(self.num_sensors)  # Default to 1500 nm

        # Main Tab Widget
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Create Tabs
        self.data_collection_tab = QWidget()
        self.file_handling_tab = QWidget()
        
        self.tabs.addTab(self.data_collection_tab, "📡 Data Collection")
        self.tabs.addTab(self.file_handling_tab, "📂 File Handling")

        # Setup Tab Layouts
        self.setup_data_collection_tab()
        self.setup_file_handling_tab()

    def setup_data_collection_tab(self):
        layout = QHBoxLayout()  # Horizontal layout to split plots and sensor info

        # Left Panel - Sensor Table and Start Button
        left_panel = QVBoxLayout()
        self.sensor_table = QTableWidget(self.num_sensors, 3)
        self.sensor_table.setHorizontalHeaderLabels(["Sensor", "Status", "CWL (nm)"])
        self.sensor_table.setFixedWidth(315)  # Ensure full width for table

        # 🔹 Automatically adjust height based on sensor count
        self.sensor_table.setFixedHeight(min(50 + self.num_sensors * 30, 315))

        # Populate table with sensors and add dropdowns for wavelength selection
        self.wavelength_dropdowns = []
        for i in range(self.num_sensors):
            self.sensor_table.setItem(i, 0, QTableWidgetItem(f"FBG {i+1}"))
            self.sensor_table.setItem(i, 1, QTableWidgetItem("🟢 Active"))
            
            # Dropdown for selecting center wavelength
            wavelength_dropdown = QComboBox()
            wavelength_dropdown.addItems([str(wl) for wl in range(1500, 1600, 5)])
            wavelength_dropdown.setCurrentText(str(self.central_wavelengths[i]))  # Load saved value
            self.sensor_table.setCellWidget(i, 2, wavelength_dropdown)
            self.wavelength_dropdowns.append(wavelength_dropdown)
        
        left_panel.addWidget(self.sensor_table)
        left_panel.addStretch(1) 

        self.save_button = QPushButton("Save Sensor Configuration")
        self.save_button.setStyleSheet("font-size: 16px; padding: 10px; height: 50px;")
        self.save_button.clicked.connect(self.save_sensor_config)
        left_panel.addWidget(self.save_button)

        self.metadata_label = QLabel("Sample #: 0\nFile Size: 0 KB\nElapsed Time: 0s\nDate: --/--/----")
        self.metadata_label.setStyleSheet("font-size: 16px; font-weight: bold;")

        left_panel.addWidget(self.metadata_label)
        left_panel.addStretch(1) 
        
        # Status Indicator
        self.status_label = QLabel("🔴 Stopped")
        self.status_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        left_panel.addWidget(self.status_label)
        left_panel.addStretch(1)

        # Start Button
        self.start_button = QPushButton("▶ START Streaming")
        self.start_button.setStyleSheet("font-size: 16px; padding: 10px; height: 50px; border: 2px solid green; color: green;")
        self.start_button.clicked.connect(self.start_live_plot)
        left_panel.addWidget(self.start_button)

        # Stop Button
        self.stop_button = QPushButton("STOP Streaming")
        self.stop_button.setStyleSheet("font-size: 16px; padding: 10px; height: 50px; border: 2px solid red; color: red;")        
        self.stop_button.clicked.connect(self.stop_live_plot)
        left_panel.addWidget(self.stop_button)


        layout.addLayout(left_panel, 1)  # Ensure left panel width respects table width

        # Right Panel - Sensor Plots
        right_panel = QVBoxLayout()
        self.plot_widget = pg.GraphicsLayoutWidget()
        self.sensor_plots = []
        self.curves = []  # Store plot curves for updating

        #max_columns = self.num_sensors%max_columns+1  # Set max columns for plots
       

        for i in range(self.num_sensors):
            column = 0
            column += i//4
            plot = self.plot_widget.addPlot(row=i%4, col=column)
            plot.setTitle(f"FBG Sensor {i+1}")
            plot.setLabel("left", "Wavelength (nm)")
            plot.showGrid(x=True, y=True)
            plot.setFixedHeight(180)  # Increase height
            plot.setFixedWidth(500)  # Set fixed width
            plot.hideAxis("bottom")  # Remove x-axis labels

            #center_wl = int(self.sensor_table.cellWidget(i, 2).currentText())  # Get selected wavelength
            #plot.setYRange(center_wl - 5, center_wl + 5)  # Apply ±5 nm limits


            curve = plot.plot(pen=pg.mkPen(color=(i*50, 100, 255), width=2))  # Create curve
            self.curves.append(curve)
            self.sensor_plots.append(plot)
        
        right_panel.addWidget(self.plot_widget)
        layout.addLayout(right_panel, 2)  # Ensure right panel width adjusts to plots

        self.data_collection_tab.setLayout(layout)

    def start_live_plot(self):
        """Start the background thread to receive live data."""
        if not self.is_active:

            # Extract the selected central wavelengths from the dropdowns
            self.central_wavelengths = [
                int(self.sensor_table.cellWidget(i, 2).currentText()) for i in range(self.num_sensors)
            ]
            print(f"Selected central wavelengths: {self.central_wavelengths}")  # Debugging output

            # Apply Y-axis limits based on selected wavelengths
            for i in range(self.num_sensors):
                
                self.sensor_plots[i].setYRange(self.central_wavelengths[i] - 1, self.central_wavelengths[i] + 1)

            self.is_active = True
            self.status_label.setText("🟢 Active")
            self.time_data = []  # Store timestamps for x-axis
            self.sensor_data = [[] for _ in range(self.num_sensors)]  # Store sensor readings

            # Initialize the data update thread
            self.data_thread = DataUpdateThread()
            self.data_thread.data_updated.connect(self.update_plot)  # Connect signal to UI update
            self.data_thread.start()  # Start the thread

    def stop_live_plot(self):
        """Stop the background thread and data streaming."""
        if self.is_active and hasattr(self, 'data_thread'):
            self.is_active = False
            self.status_label.setText("🔴 Stopped") 
            self.data_thread.terminate()


    def update_plot(self, df):
        """Update the plots with new sensor data."""
        self.time_data.append(time.time())  # Append new timestamp

        for i in range(self.num_sensors):
            self.sensor_data[i].append(df["Wavelength (nm)"][i])  # Append new data point
            
            # Keep only the last 50 points for smooth plotting
            if len(self.sensor_data[i]) > 50:
                self.sensor_data[i].pop(0)

        # Update plots dynamically
        for i in range(self.num_sensors):
            self.curves[i].setData(self.time_data[-50:], self.sensor_data[i])

    def setup_file_handling_tab(self):
        layout = QVBoxLayout()
        self.file_handling_tab.setLayout(layout)

    def load_sensor_config(self):
        """Load sensor configuration from a JSON file."""
        try:
            with open(CONFIG_FILE, "r") as f:
                config = json.load(f)
                return config.get("central_wavelengths", [1500] * self.num_sensors)
        except (FileNotFoundError, json.JSONDecodeError):
            return [1500] * self.num_sensors  # Default to 1500 nm

    def save_sensor_config(self):
        """Save the current sensor configuration to a JSON file."""
        self.central_wavelengths = [
            int(self.sensor_table.cellWidget(i, 2).currentText()) for i in range(self.num_sensors)
        ]
        config = {"central_wavelengths": self.central_wavelengths}
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f)
        print("Configuration saved:", config)  # Debugging output


# Run the App
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HyperionDAQUI()
    window.show()
    sys.exit(app.exec())