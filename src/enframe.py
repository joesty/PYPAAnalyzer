import numpy as np

def enframe(x, win=None, hop=None, m=None, fs=None):
    nx = len(x)
    if win is None:
        win = nx
    if m is None:
        m = ''
    #print(type(m))
    nwin = len(win) if hasattr(win, '__len__') else 1
    if nwin == 1:
        lw = win
        w = np.ones((1, lw))
    else:
        lw = nwin
        w = win.T
    if hop is None:
        hop = lw
    elif hop < 1:
        hop = int(lw * hop)
    if 'a' in m:
        w = w * np.sqrt(hop / np.sum(w**2))
    elif 's' in m:
        w = w / np.sqrt(np.sum(w**2) * lw)
    elif 'S' in m:
        w = w / np.sqrt(np.sum(w**2) * lw / hop)
    if 'd' in m:
        if fs is None:
            w = w * np.sqrt(lw)
        else:
            w = w * np.sqrt(lw / fs)
    nli = nx - lw + hop
    nf = max(int(nli / hop), 0)
    na = nli - hop * nf + (nf == 0) * (lw - hop)
    fx = m is not None and (('z' in m) or ('r' in m)) and na > 0
    f = np.zeros((nf + fx, lw))
    indf = hop * np.arange(nf).reshape(-1, 1)
    inds = np.arange(lw)
    if fx:
        f[:nf, :] = x[indf + inds]
        if 'r' in m:
            ix = 1 + np.mod(np.arange(nf * hop, nf * hop + lw), 2 * nx)
            f[nf, :] = x[ix + (ix > nx) * (2 * nx + 1 - 2 * ix)]
        else:
            f[nf, :nx - nf * hop] = x[nf * hop:nx]
        nf = f.shape[0]
    else:
        f[:, :] = x[indf + inds]
    if nwin > 1:
        f = f * w
    if 'p' in m:
        f = np.fft.fft(f, axis=1)
        f = np.real(f * np.conj(f))
        if 'p' in m:
            imx = int(np.fix((lw + 1) / 2))
            f[:, 1:imx] = f[:, 1:imx] + f[:, lw:0:-1]
            f = f[:, :int(np.fix(lw / 2) + 1)]
    elif 'f' in m:
        f = np.fft.fft(f, axis=1)
        if 'f' in m:
            f = f[:, :int(np.fix(lw / 2)) + 1]
    if 'E' in m:
        t0 = np.sum((np.arange(lw) + 1) * w**2) / np.sum(w**2)
    elif 'A' in m:
        t0 = np.sum((np.arange(lw) + 1) * w) / np.sum(w)
    else:
        t0 = (1 + lw) / 2
    t = t0 + hop * np.arange(nf)
    return f, t, w
