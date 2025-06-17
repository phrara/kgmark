import numpy as np
from scipy.fftpack import dct, idct
from .quantify import Quantify

def QIMDehide(I, delta, length):
    block = (8, 8)
    si = I.shape

    # Determine the number of blocks in rows and columns
    N = si[1] // block[1]  # Number of blocks per row
    M = si[0] // block[0]  # Number of blocks per column

    # Initialize output data array
    o = np.zeros(M * N, dtype=int)
    idx = 0

    for i in range(M):
        rst, red = i * block[0], (i + 1) * block[0]

        for j in range(N):
            cst, ced = j * block[1], (j + 1) * block[1]
            tmp = I[rst:red, cst:ced].astype(np.float32)

            # Apply DCT
            tmp = dct(dct(tmp.T, norm='ortho').T, norm='ortho')

            # Extract bits from diagonal coefficients
            to = np.zeros(block[0], dtype=int)
            for k in range(block[0]):
                l = block[0] - 1 - k  # Calculate diagonal position
                q00 = Quantify(tmp[k, l], 0, delta)
                q10 = Quantify(tmp[k, l], 1, delta)

                # Determine the closest quantized value
                pos = np.argmin(np.abs(tmp[k, l] - np.array([q00, q10])))
                to[l] = pos
            # print(to)
            # Decide the bit value for this block
            o[idx] = 255 if np.sum(to) >= 4 else 0
            idx += 1

    # Return the extracted data limited to the specified length
    return o[:length]