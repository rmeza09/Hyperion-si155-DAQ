from hyperion import Hyperion
from socket import gaierror
import numpy as np
class Interrogator():
    def __init__(self, address, timeout: float = 1):
        self.address = address
        self.is_connected = False
        self.signal_each_ch = np.zeros(4) #number of channels
        self.total_reading_num = 0
        self.available_ch = {}
        self.connect()
        self.check_ch_available()

    def connect( self ):
        self.interrogator = Hyperion( self.address )
        try:
            self.is_ready = self.interrogator.is_ready
            self.is_connected = True
            self.num_chs = self.interrogator.channel_count
            print("interrogator connected")
        except OSError:
            self.is_connected = False
            print("Fail to connect with Interrogator!")

    def check_ch_available(self):
        peaks = self.interrogator.peaks
        for idx in range(1, self.num_chs+1):
            numCHSensors = np.size(peaks[idx].astype( np.float64 ))
            self.available_ch['CH1'] = numCHSensors
            self.signal_each_ch[idx-1] = numCHSensors
            self.total_reading_num += numCHSensors

    def getData(self) -> np.ndarray:
        peaks = self.interrogator.peaks
        raw_data = []
        for idx in range( 1, self.num_chs+1 ):
            raw_data = np.concatenate((raw_data,peaks[idx].astype( np.float64 )))
        # for
        return raw_data

def main( args=None):
    interrogator = Interrogator("10.0.0.55")
    print(interrogator.interrogator.is_ready)
    print(interrogator.is_connected)
    print(interrogator.num_chs)
    peaks = interrogator.interrogator.peaks
    print(interrogator.getData())
    print(interrogator.signal_each_ch)
    print(interrogator.total_reading_num)




if __name__ == "__main__":
    main()



