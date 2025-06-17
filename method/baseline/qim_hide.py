import numpy as np
from scipy.fftpack import dct, idct

from .quantify import Quantify

def QIMHide(I, data, delta):
    block = [8, 8]
    si = I.shape
    lend = len(data)
    N = int(np.floor(si[1] / block[1]))  
    M = min(int(np.floor(si[0] / block[0])), int(np.ceil(lend / N)))
    if lend < M * N:
        data = np.concatenate((data, np.zeros(M * N - lend, dtype=data.dtype)))

    o = I.copy()
    idx = 0

    
    for i in range(M):
        rst, red = i * block[0], (i + 1) * block[0]

        for j in range(N):
            cst, ced = j * block[1], (j + 1) * block[1]
            tmp = I[rst:red, cst:ced].astype(np.float32)

            # Apply DCT
            tmp = dct(dct(tmp.T, norm='ortho').T, norm='ortho')

            # Modify diagonal coefficients
            _data = 1 if data[idx] > 0 else 0
            for k in range(block[0]):
                l = block[0] - 1 - k  # Calculate diagonal position
                tmp[k, l] = Quantify(tmp[k, l], _data, delta)

            # Apply inverse DCT
            tmp = idct(idct(tmp.T, norm='ortho').T, norm='ortho')

            # Assign modified block back to the output image
            o[rst:red, cst:ced] = tmp
            idx += 1

    return o