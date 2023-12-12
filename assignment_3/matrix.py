import numpy as np


class Matrix:

    def __init__(self, s, PLY, PY, PnLnY):
        self.s = s
        self.PLY = PLY
        self.PY = PY
        self.PnLnY = PnLnY
        self.PnLY = 1 - PLY
        self.PnY = 1 - PY

    @property
    def heights(self):
        PnLY = self.PnLY
        PY = self.PY
        PLY = self.PLY
        s = self.s
        PnLnY = self.PnLnY
        PnY = self.PnY

        matrix = np.array([
            [1, pow(PY, 4), 1, pow(PnLY, 7), 1, 1],  # Phi 1
            [1, pow(PY, 3), 1, pow(PnLY, 6), pow(s, 1), pow(((PLY * PY) + (PnLnY * PnY)), 1)],  # Phi 2
            [1, pow(PY, 2), 1, pow(PnLY, 5), pow(s, 2), pow(((PLY * PY) + (PnLnY * PnY)), 2)],  # Phi 3
            [1, pow(PY, 1), 1, pow(PnLY, 4), pow(s, 3), pow(((PLY * PY) + (PnLnY * PnY)), 3)],  # Phi 4
            [1, 1, 1, pow(PnLY, 3), pow(s, 4), pow(((PLY * PY) + (PnLnY * PnY)), 4)],  # Phi 5
            [1, 1, pow(PLY, 1), pow(PnLY, 2), pow(s, 5), pow(((PLY * PY) + (PnLnY * PnY)), 4)],  # Phi 6
            [1, 1, pow(PLY, 2), pow(PnLY, 1), pow(s, 6), pow(((PLY * PY) + (PnLnY * PnY)), 4)],  # Phi 7
            [1, 1, pow(PLY, 3), 1, pow(s, 7), pow(((PLY * PY) + (PnLnY * PnY)), 4)]  # Phi 8
        ])

        a = 1 / np.sum([np.prod(x) for x in matrix])

        for i, x in enumerate(matrix):
            matrix[i][0] = a

        heights = [np.prod(x) for x in matrix]
        return heights
