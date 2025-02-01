from scipy.signal import convolve, lfilter, butter, sosfilt # type: ignore
from shiftcov import shiftcov
from AdaptLevel import AdaptLevel
from shiftcov import shiftcov
import numpy as np
from scipy.signal import convolve, lfilter # type: ignore
from scipy.io import loadmat #type: ignore
from ERBFilterBank import ERBFilterBank
from AdaptLevel import AdaptLevel

def digitalFilter(coefs, data):
    filtered_signal = sosfilt(coefs, data)
    return filtered_signal

def FAnalyzer(xr):
    if np.sum(np.abs(xr)) == 0:
        return 0
    data = loadmat('libF/Rpar.mat')
    coeficients = np.array([
    [0.00148372763971299, 0, -0.00148372763971299, 1, -1.99890021308069, 0.998910140107264],
    [0.00148372763971299, 0, -0.00148372763971299, 1, -1.99993224055117, 0.999932278850526],
    [0.00148293492832470, 0, -0.00148293492832470, 1, -1.99690080716171, 0.996910166492610],
    [0.00148293492832470, 0, -0.00148293492832470, 1, -1.99979638333837, 0.999796424058798],
    [0.00148232907092239, 0, -0.00148232907092239, 1, -1.99966449031569, 0.999664535462904],
    [0.00148232907092239, 0, -0.00148232907092239, 1, -1.99540152000579, 0.995409956439619],
    [0.00148200149278950, 0, -0.00148200149278950, 1, -1.99956588364273, 0.999565933360383],
    [0.00148200149278950, 0, -0.00148200149278950, 1, -1.99461844932247, 0.994626084188995]
    ])
    # Check available keys in the data dictionary
    #print(data.keys())

    # Extract each variable
    a = data['a'].flatten()
    b = data['b'].flatten()
    c = data['c'].flatten()
    d = data['d'].flatten()
    Nch = data['Nch'].flatten()[0]
    fcoefs = data['fcoefs']
    gzi = data['gzi'].flatten()
    fs = data['fs'].flatten()[0]
    xr = xr.T
    xr = AdaptLevel(xr, 60)
    conv_bd = convolve(b, d)
    conv_ac = convolve(a, c)

    signal = lfilter(conv_bd, conv_ac, xr)


    exc = ERBFilterBank(signal, fcoefs)
    exc = np.maximum(exc, 0)
    etmp = np.abs(exc)

    aa = np.ones([1, exc.shape[1]])
    bb = np.mean(etmp, axis=1)
    h0 = np.kron(aa, bb).T.reshape(exc.shape[1], exc.shape[0]).T
    excd = etmp - h0
    hBPi = np.zeros([Nch, excd.shape[1]])
    for i in range(Nch):
        hBPi[i, :] = digitalFilter(coeficients, excd[i, :])
    #print(hBPi[:, 0])
    hBPrms = np.sqrt(np.mean(hBPi**2, axis=1))
    rexc = np.sqrt(np.mean(exc**2, axis=1))

    ##daqui pra baixo pode ta errado

    # Supondo que rexc, h0, hBPrms, hBPi, gzi, fs, Nch são arrays numpy ou variáveis definidas anteriormente

    maxi = np.max(rexc)
    if maxi > 0:
        calib = rexc / maxi
    else:
        calib = 0

    mdepth = np.zeros(int(Nch))

    for n in range(Nch):
        if h0[n, 0] > 0:
            mdepth[n] = hBPrms[n] / h0[n, 0]
            mdepth[n] *= calib[n]
        else:
            mdepth[n] = 0

    ki = np.zeros(Nch)
    for n in range(Nch):
        if n < Nch - 2:
            amount = 0.003 * fs
            ki[n] = shiftcov(hBPi[n, :], hBPi[n + 2, :], amount)

    #print(ki)

    Cf = 0.3423
    fi = np.zeros(Nch)
    fi[0:2] = (gzi[0:2] * mdepth[0:2] * ki[0:2])**2
    fi[2:Nch-2] = (gzi[2:Nch-2] * mdepth[2:Nch-2] * ki[2:Nch-2] * ki[0:Nch-4])**2
    fi[Nch-2:Nch] = (gzi[Nch-2:Nch] * mdepth[Nch-2:Nch] * ki[Nch-2:Nch])**2
    F = Cf * np.sum(fi)
    return F