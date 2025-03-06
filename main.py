from frontend.main_UI import HyperionDAQUI
from backend.si155_read import Interrogator

from PyQt6.QtWidgets import QApplication
import sys

def main():
    ip_address = "10.0.0.55"
    interrogator = Interrogator(ip_address)  # Direct initialization

    app = QApplication(sys.argv)
    window = HyperionDAQUI(interrogator)  # Pass interrogator directly
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
