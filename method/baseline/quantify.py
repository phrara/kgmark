import numpy as np


def Quantify(x, num, delta):
    dither = (num * 2 - 1) * (delta / 4)  # '(num * 2 - 1)' makes '0,1' to '-1, +1'
    y = delta * np.round((x - dither) / delta) + dither
    return y