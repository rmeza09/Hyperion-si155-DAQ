import sys
import os

from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QTableWidget, QTableWidgetItem, QLabel, QPushButton, QComboBox, QListWidget
from PyQt6.QtCore import QThread, pyqtSignal, Qt
import pyqtgraph as pg
import pandas as pd
import time
import numpy as np
import json

# Dynamically locate the backend folder and add it to sys.path
backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend"))
if backend_path not in sys.path:
    sys.path.append(backend_path)

# Now backend modules can be imported normally
from hdf5_to_csv import convert_hdf5_to_csv

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
    def __init__(self, NumSensors):
        super().__init__()

        # Window Settings
        self.setWindowTitle("Hyperion SI155 DAQ")
        self.setGeometry(500, 100, 1400, 800)  # Adjusted for better layout

        # Hardcoded number of sensors (for now, later will be dynamic)
        self.num_sensors = NumSensors  # This will later be replaced with a backend call
        self.is_active = False

        self.central_wavelengths = self.load_sensor_config()  # Default to 1500 nm

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
        

        self.save_button = QPushButton("Save Sensor Configuration")
        self.save_button.setStyleSheet("font-size: 16px; padding: 10px; height: 50px;")
        self.save_button.clicked.connect(self.save_sensor_config)
        left_panel.addWidget(self.save_button)
        left_panel.addStretch(1) 

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
            plot.setTitle(f"FBG Sensor: {self.central_wavelengths[i]} nm")
            plot.setLabel("left", "Wavelength [nm]")
            plot.showGrid(x=True, y=True)
            plot.setFixedHeight(180)  # Increase height
            plot.setFixedWidth(500)  # Set fixed width
            plot.hideAxis("bottom")  # Remove x-axis labels

            curve = plot.plot(pen=pg.mkPen(color=(i*50, 100, 255), width=2))  # Create curve
            self.curves.append(curve)
            self.sensor_plots.append(plot)
        
        right_panel.addWidget(self.plot_widget)
        layout.addLayout(right_panel, 2)  # Ensure right panel width adjusts to plots

        self.data_collection_tab.setLayout(layout)

    def update_plot_titles(self):
        """ Updates the plot titles dynamically based on the saved central wavelengths. """
        for i, plot in enumerate(self.sensor_plots):
            plot.setTitle(f"FBG Sensor: {self.central_wavelengths[i]} nm")


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
                
                self.sensor_plots[i].setYRange(self.central_wavelengths[i] - 0.5, self.central_wavelengths[i] + 0.5)

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
        """Setup the File Handling tab to display collected data files and metadata."""
        layout = QHBoxLayout()  # Horizontal layout for file list and metadata

        # Left Column - File List
        left_panel = QVBoxLayout()
        self.file_list = QListWidget()
        self.file_list.itemClicked.connect(self.display_file_metadata)  # Event to show metadata
        label1 = QLabel("📂 Collected HDF5 Files:")
        left_panel.addWidget(label1)
        label1.setStyleSheet("font-weight: bold; font-size: 16px;")
        left_panel.addWidget(self.file_list)
        
        # CSV File List
        self.csv_list = QListWidget()
        label2 = QLabel("📂 Converted CSV Files:")
        left_panel.addWidget(label2)
        label2.setStyleSheet("font-weight: bold; font-size: 16px;")
        left_panel.addWidget(self.csv_list)
        
        layout.addLayout(left_panel, 1)

        # Refresh Button
        self.refresh_button = QPushButton("🔄 Refresh File List")
        self.refresh_button.setStyleSheet("font-size: 16px; padding: 8px; height: 50px;")
        self.refresh_button.clicked.connect(lambda: [self.update_file_list(), self.update_csv_list()])
        left_panel.addWidget(self.refresh_button)
        
        # Right Column - File Metadata and Conversion Button
        right_panel = QVBoxLayout()
        right_panel.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.file_metadata_label = QLabel("Select a file to view metadata.")
        self.file_metadata_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        label3 = QLabel("📄 File Conversion from HDF5 to CSV:")
        right_panel.addWidget(label3)
        label3.setStyleSheet("font-weight: bold; font-size: 16px; padding-bottom: 20px;")
        self.file_metadata_label.setStyleSheet("font-size: 16px; padding-bottom: 20px;")
        right_panel.addWidget(self.file_metadata_label)
        
        # Convert to CSV Button
        self.convert_button = QPushButton("📄 Convert to CSV")
        self.convert_button.setStyleSheet("font-size: 16px; padding: 8px; height: 50px;")
        self.convert_button.clicked.connect(self.convert_selected_hdf5)
        right_panel.addWidget(self.convert_button)
        
        layout.addLayout(right_panel, 1)

        # Populate file lists on startup
        self.update_file_list()
        self.update_csv_list()
        self.file_handling_tab.setLayout(layout)

    def update_file_list(self):
        """Fetch and display all HDF5 files in the DATA folder."""
        self.file_list.clear()
        data_folder = "DATA"
        
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)  # Create the folder if it doesn't exist

        files = [f for f in os.listdir(data_folder) if f.endswith(".h5")]
        
        if files:
            self.file_list.addItems(files)
        else:
            self.file_list.addItem("No files found")

    def update_csv_list(self):
        """Fetch and display all CSV files in the DATA/CSV folder."""
        self.csv_list.clear()
        csv_folder = "DATA/CSV"
        
        if not os.path.exists(csv_folder):
            os.makedirs(csv_folder)  # Create the CSV folder if it doesn't exist

        files = [f for f in os.listdir(csv_folder) if f.endswith(".csv")]
        
        if files:
            self.csv_list.addItems(files)
        else:
            self.csv_list.addItem("No CSV files found")


    def display_file_metadata(self, item):
        """Display metadata for the selected HDF5 file."""
        file_path = os.path.join("DATA", item.text())
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path) // 1024  # Convert bytes to KB
            modified_time = time.strftime('%m/%d/%Y %I:%M %p', time.localtime(os.path.getmtime(file_path)))
            
            metadata_text = (f"File: {item.text()}\n"
                            f"Size: {file_size} KB\n"
                            f"Last Modified: {modified_time}")
            self.file_metadata_label.setText(metadata_text)

    def convert_selected_hdf5(self):
        """Convert the selected HDF5 file to CSV and store it in the CSV folder."""
        selected_item = self.file_list.currentItem()
        if not selected_item:
            self.file_metadata_label.setText("No file selected for conversion.")
            return
        
        hdf5_filepath = os.path.join("DATA", selected_item.text())
        csv_folder = "DATA/CSV"
        if not os.path.exists(csv_folder):
            os.makedirs(csv_folder)
        
        # Generate CSV file path
        csv_filepath = os.path.join(csv_folder, selected_item.text().replace(".h5", ".csv"))
        
        # Convert the file using the existing backend function
        #from backend.hdf5_to_csv import convert_hdf5_to_csv
        convert_hdf5_to_csv(hdf5_filepath)
        
        # Move the generated CSV file to the CSV folder
        if os.path.exists(hdf5_filepath.replace(".h5", ".csv")):
            os.rename(hdf5_filepath.replace(".h5", ".csv"), csv_filepath)
        
        self.file_metadata_label.setText(f"Converted and saved: {csv_filepath}")
        self.update_csv_list()  # Refresh the CSV file list


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
        self.update_plot_titles()  # Update plot titles with new wavelengths
        print("Configuration saved:", config)  # Debugging output


# Run the App
if __name__ == "__main__":
    NumSensors = 5
    app = QApplication(sys.argv)
    window = HyperionDAQUI(NumSensors)
    window.show()
    sys.exit(app.exec())