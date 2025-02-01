from shiftcov import shiftcov
import numpy as np
from scipy.signal import convolve, lfilter # type: ignore
from scipy.io import loadmat #type: ignore
from enframe import enframe
from ERBFilterBank import ERBFilterBank
from AdaptLevel import AdaptLevel

def RAnalyzer(xr):
    if np.sum(np.abs(xr)) == 0:
        return 0
    filt = {
    "numH7": np.array([-9.214e-07, 9.828e-07, 2.952e-07, -2.868e-07, -6.984e-08], dtype=np.float128),
    "denH7": np.array([1.0000, -5.8996, 14.5136,  -19.0573, 14.0867, -5.5577,  0.9143], dtype=np.float128),
    "numH14": np.array([0.0003, -0.0009,  0.0011, -0.0006, 0.0001]),
    "denH14": np.array([1.0000, -5.2990, 11.6705, -13.6729, 8.9869, -3.1422, 0.4567]),
    "numH30": np.array([1.29727267205170e-05, -1.47102163717761e-05, 1.16924135642150e-06, -3.85658885357684e-06, 4.42843669156559e-06]),
    "denH30": np.array([1, -5.80122166192452, 14.0537019475717, -18.1982190479573, 13.2849820966997, -5.18399725914718, 0.844754071932504]),
    "numH36": np.array([3.05545894165424e-05, -4.82784630733298e-05, 1.77214106193544e-05, 5.39435230870577e-09, 8.15705520793788e-11]),
    "denH36": np.array([1.0000, -5.7004, 13.5701, -17.2687, 12.3899, -4.7523, 0.7613]),
    "numH66": np.array([1.49958281527685e-05, -4.25867689844540e-06, -1.10078886192767e-05, 1.57882602845410e-07, 1.28586558461309e-07]),
    "denH66": np.array([1.0000, -5.4004, 12.1138, -14.4355, 9.6287, -3.4041, 0.4975])
    }


    numH7 = np.array([-9.21451216121396e-07, 9.82888560318938e-07, 2.95260688176911e-07, -2.86857741098071e-07, -6.98406897642615e-08])
    denh7 = np.array([1, -5.8996172188963, 14.5135559959204, -19.0572652130071, 14.0866687009398, -5.55767912464569, 0.91433686300854])
    numh14 = np.array([0.000306736050581849, -0.000939253934583252, 0.00110641981400735, -0.000593687072990132, 0.000119785307608896])
    denh14 = np.array([1, -5.29899326045918, 11.6704997567747, -13.6728690880423, 8.98692026939131, -3.14222989772337, 0.456672622395994])
    numh30 = np.array([1.2972726720517e-05, -1.47102163717761e-05,   1.1692413564215e-06, -3.85658885357684e-06,  4.42843669156559e-06])
    denh30 = np.array([1, -5.80122166192452,  14.0537019475717, -18.1982190479573,  13.2849820966997, -5.18399725914718, 0.844754071932504])
    numh36 = np.array([3.05545894165424e-05, -4.82784630733298e-05,  1.77214106193544e-05,  5.39435230870577e-09,  8.15705520793788e-11])
    denh36 = np.array([1, -5.70038528212244,  13.5700852476812, -17.2686675660105,  12.3899328214928, -4.75225971584698, 0.761294748827704])
    numh66 = np.array([1.49958281527685e-05, -4.2586768984454e-06, -1.10078886192767e-05,  1.5788260284541e-07, 1.28586558461309e-07])
    denh66 = np.array([ 1, -5.40036700642642,  12.1137600502307, -14.4354847038082,  9.62867730590816, -3.40407176241577, 0.497486522472457])

    filt = {
        "numH7": numH7,
        "denH7": denh7,
        "numH14": numh14,
        "denH14": denh14,
        "numH30": numh30,
        "denH30": denh30,
        "numH36": numh36,
        "denH36": denh36,
        "numH66": numh66,
        "denH66": denh66
    }
    
    data = loadmat('libR/Rpar.mat')

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
    xr
    winlen = 1 ##pode ter erro aqui a implementação original é: size(xr,1)
    wlen = winlen*fs
    xf, _, _ = enframe(xr, np.blackman(wlen))
    conv_bd = convolve(b, d)
    conv_ac = convolve(a, c)

    signal = lfilter(conv_bd, conv_ac, xf)

    #print('signal shape:', signal.shape)

    exc = ERBFilterBank(signal, fcoefs)
    exc = np.maximum(exc, 0)
    etmp = np.abs(exc)

    aa = np.ones([1, exc.shape[1]])
    bb = np.mean(etmp, axis=1)
    h0 = np.kron(aa, bb).T.reshape(exc.shape[1], exc.shape[0]).T
    excd = etmp - h0
    hBPi = np.zeros_like(excd)
    hBPi[:11, :] = lfilter(b=filt['numH7'], a=filt['denH7'], x=excd[:11,:], axis=1)
    hBPi[11:28, :] = lfilter(b=filt['numH14'], a=filt['denH14'], x=excd[11:28,:], axis=1)
    hBPi[28:36, :] = lfilter(b=filt['numH30'], a=filt['denH30'], x=excd[28:36,:], axis=1)
    hBPi[36:65, :] = lfilter(b=filt['numH36'], a=filt['denH36'], x=excd[36:65,:], axis=1)
    hBPi[65:, :] = lfilter(b=filt['numH66'], a=filt['denH66'], x=excd[65:,:], axis=1)
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

    Cr = 1.1998
    ri = np.zeros(Nch)
    ri[0:7] = (gzi[0:7] * mdepth[0:7] * ki[0:7])**2
    ri[7:Nch-3] = (gzi[7:Nch-3] * mdepth[7:Nch-3] * ki[5:Nch-5] * ki[7:Nch-3])**2
    ri[Nch-3] = (gzi[Nch-3] * mdepth[Nch-3] * ki[Nch-5])**2
    ri[Nch-2] = (gzi[Nch-2] * mdepth[Nch-2] * ki[Nch-4])**2
    ri[Nch-1] = (gzi[Nch-1] * mdepth[Nch-1] * ki[Nch-3])**2
    R = Cr * np.sum(ri)
    return R
