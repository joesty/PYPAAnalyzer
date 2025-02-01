import numpy as np

def AdaptLevel(xr, level):
    cols = len(xr)
    rmsSig = np.sqrt(np.sum(xr**2, axis=0)/cols)
    factor = 10**((level-30)/20 - np.log10(rmsSig))
    ###pode haver um erro aqui
    return xr*factor