import os
import numpy as np


class Calibration:
    def __init__(self, calibration_path):
        self.calibration_path = calibration_path
        self.calibration_matrix = {}

        with open(os.path.join(calibration_path), 'r') as calibration_file:
            for line in calibration_file:
                line = line.split()
                try:
                    self.calibration_matrix[line[0][:-1]] = np.array(line[1:], dtype=np.float32)
                except ValueError:
                    continue

        self.P0 = self.calibration_matrix["P0"]
        self.P1 = self.calibration_matrix["P1"]
        self.P2 = self.calibration_matrix["P2"].reshape(3, 4)
        self.P3 = self.calibration_matrix["P3"]
        self.R0_rect = self.calibration_matrix["R0_rect"].reshape(3, 3)
        self.Tr_velo_to_cam = self.calibration_matrix["Tr_velo_to_cam"].reshape(3, 4)
