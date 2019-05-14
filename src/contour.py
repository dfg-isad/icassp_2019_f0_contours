import numpy as np


class Contour:
    """ Class implementation of single frequency contour"""
    # todo add magnitude contour parameter
    def __init__(self,
                 f0_hz,
                 t_sec,
                 f0_cent_rel):
        self.f0_hz = f0_hz
        self.t_sec = t_sec
        self.f0_cent_rel = f0_cent_rel
        assert len(self.f0_hz) == len(self.t_sec) == len(self.f0_cent_rel)

    def __len__(self):
        return len(self.f0_hz)

    @staticmethod
    def hz_to_cent(f_hz, f_ref_hz):
        """ Convert absolute frequencies to frequency deviations in cent
        Args:
            f_hz (np.ndarray): Absolute frequency values in Hz
            f_ref_hz (float): Reference frequency (e.g. from MIDI pitch
                        f = 440*2**((f-69)/12)
        Returns:
            f_rel_cent (np.ndarray): Frequency deviations from reference frequency in cent
        """
        return 1200*np.log2(f_hz/f_ref_hz)
