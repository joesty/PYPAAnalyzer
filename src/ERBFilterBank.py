from scipy.signal import lfilter #type: ignore
import numpy as np

def ERBFilterBank(signal, fcoefs):
    A0 = fcoefs[:, 0]
    A11 = fcoefs[:, 1]
    A12 = fcoefs[:, 2]
    A13 = fcoefs[:, 3]
    A14 = fcoefs[:, 4]
    A2 = fcoefs[:, 5]
    B0 = fcoefs[:, 6]
    B1 = fcoefs[:, 7]
    B2 = fcoefs[:, 8]
    gain = fcoefs[:, 9]
    output = np.zeros([gain.shape[0], max(signal.shape)])
    for chan in range(gain.shape[0]):
        y1 = lfilter(
            [A0[chan]/gain[chan], A11[chan]/gain[chan], A2[chan]/gain[chan]],
            [B0[chan], B1[chan], B2[chan]],
            signal
        )
        y2 = lfilter(
            [A0[chan], A12[chan], A2[chan]],
            [B0[chan], B1[chan], B2[chan]],
            y1
        )
        y3 = lfilter(
            [A0[chan], A13[chan], A2[chan]],
            [B0[chan], B1[chan], B2[chan]],
            y2
        )
        y4 = lfilter(
            [A0[chan], A14[chan], A2[chan]],
            [B0[chan], B1[chan], B2[chan]],
            y3
        )
        output[chan, :] = y4
    return output
