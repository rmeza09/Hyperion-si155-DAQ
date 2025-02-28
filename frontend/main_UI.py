from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QTableWidget, QTableWidgetItem, QLabel, QPushButton, QComboBox
from PyQt6.QtCore import QThread, pyqtSignal
import pyqtgraph as pg
import sys
import pandas as pd
import time
from mock_sampling import mock_sampling  # Import mock data generator

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
        for i in range(self.num_sensors):
            self.sensor_table.setItem(i, 0, QTableWidgetItem(f"FBG {i+1}"))
            self.sensor_table.setItem(i, 1, QTableWidgetItem("🟢 Active"))
            
            # Dropdown for selecting center wavelength
            wavelength_dropdown = QComboBox()
            wavelength_dropdown.addItems([str(wl) for wl in range(1500, 1600, 5)])
            self.sensor_table.setCellWidget(i, 2, wavelength_dropdown)
        
        left_panel.addWidget(self.sensor_table)
        left_panel.addStretch(1) 
        self.metadata_label = QLabel("Sample #: 0\nFile Size: 0 KB\nElapsed Time: 0s\nDate: --/--/----")
        self.metadata_label.setStyleSheet("font-size: 16px; font-weight: bold;")

        left_panel.addWidget(self.metadata_label)
        left_panel.addStretch(1) 
        
        # Start Button
        self.start_button = QPushButton("▶ Start Mock Sampling")
        self.start_button.setStyleSheet("font-size: 16px; padding: 10px; height: 50px;")  # Make button larger
        self.start_button.clicked.connect(self.start_live_plot)
        left_panel.addWidget(self.start_button)

        
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
        self.time_data = []  # Store timestamps for x-axis
        self.sensor_data = [[] for _ in range(self.num_sensors)]  # Store sensor readings

        # Initialize the data update thread
        self.data_thread = DataUpdateThread()
        self.data_thread.data_updated.connect(self.update_plot)  # Connect signal to UI update
        self.data_thread.start()  # Start the thread

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

# Run the App
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HyperionDAQUI()
    window.show()
    sys.exit(app.exec())