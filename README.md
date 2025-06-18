# Live Data Acquisition, Viewing, and File Management for Hyperion SI155
For this project I designed a desktop app for interfacing with the Hyperion sensing instrument through its Python API. 
This desktop app allows me to stream data at the maximum sampling rate while running a concurrent process which allows me to downsample data points and display a live view to confirm data collection.
I designed a configuration feature for adding a specific number of sensors and specifying the center wavelength for each sensor.
Process can be started and stopped with button control which automatically begins writing data to an HDF5 file database.

## 🧰 Built With
- Language: Python 3.9
- Frontend: PyQt6.QtWidgets, PyQt6.QtCore
- Threading/Concurrent Processes: PyQt6.QtCore import QThread
- Plotting: matplotlib, pyqtgraph
- Data handling: pandas, NumPy, scipy.signal
- File Writing: HDF5, with integrated conversion to CSV
- Micron Optics SDK, Hyperion Python API
