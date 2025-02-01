import numpy as np


def shiftcov(f1, f2, amount):
    L = len(f1)
    #print(L)
    ff2 = f2.copy()
    r = []
    for i in range(0, int(amount), 10):  # The signal is shifted 10 by 10 samples for less
        cfac = np.cov(f1, f2)
        #print(cfac)
        den = np.sqrt(np.outer(np.diag(cfac), np.diag(cfac)))
        #print("den:", den)
        if den[1, 0] > 0:
            r.append(cfac[1, 0] / den[1, 0])
        else:
            r.append(0)
        f2 = np.roll(f2, -1)
        f2[-1] = 0

    f1_shifted = f1.copy()
    for i in range(0, int(amount), 10 ):
        f1_shifted = np.concatenate((np.zeros(i+1), f1_shifted[:L-(i+1)]))
        cfac = np.cov(f1_shifted, ff2)
        #print(cfac)
        den = np.sqrt(np.outer(np.diag(cfac), np.diag(cfac)))
        #print(den)
        if den[1, 0] > 0:
            r.append(cfac[1, 0] / den[1, 0])
        else:
            r.append(0)

    y = np.max(r)

    return y
