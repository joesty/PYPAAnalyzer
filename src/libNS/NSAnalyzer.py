from enframe import enframe
import numpy as np
from DIN45631 import DIN45631
from scipy.io import loadmat
def NSAnalyzer(xr):
    if np.sum(np.abs(xr)) == 0:
        return 0, 0
    H = loadmat('data/H.mat')['H'] ##tirar daqui
    #print(xr[0])
    #print(xr[-1])
    wlen = xr.shape[0]
    Xf = np.fft.fft2(enframe(xr, np.hanning(wlen))[0].T)
    Xf = Xf[:int(wlen/2)+1, :]
    #print(Xf)

    Xf_sp1 = 88.14 + 20*np.log10(np.abs(Xf)/(wlen/4))
    #print(Xf_sp1)
    mxFlt=28
    YdBZinput = np.zeros(mxFlt)
    for i in range(0, mxFlt):
        YdBZinput[i] = 10*np.log10(np.sum((10**(Xf_sp1.T / 10))*(np.abs((H[i][:])*(H[i][:])))))
    YdBZinput
    YdBZinput[YdBZinput < -70] = -70
    N, lspec, _ = DIN45631(YdBZinput, 0)
    nl = len(lspec)
    gz = np.ones(240)
    z = np.arange(141, nl+1)
    gz[z-1] = 0.00012*(z/10)**4-0.0056*(z/10)**3+0.1*(z/10)**2-0.81*(z/10)+3.5

    z = np.arange(0.1, nl/10+0.1, 0.1)
    divisor = np.sum(lspec*0.1)
    S = 0
    if divisor != 0:
        S = 0.11 * np.sum(lspec*gz*z*0.1) / np.sum(lspec*0.1)
    return float(N), float(S)