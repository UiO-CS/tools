'''Implements wrappers around pywavelets to create multi-level wavelet
transforms

NOTE: What pywavelets call cV and cH is reversed from what is typical. These functions will use
+--------+--------+
|        |        |
|   cA   |   cH   |
|        |        |
+--------+--------+
|        |        |
|   cV   |   cD   |
|        |        |
+--------+--------+
'''
import pywt

def dwt2(z, wavelet, levels=1, mode='periodization'):
    if levels == 0:
        return z

    cA, (cV, cH, cD) = pywt.dwt2(z, wavelet, mode)
    return np.block([[dwt2(cA, levels-1), cH], [cV, cD]])

def idwt2(z, wavelet, levels=1, mode='periodization'):
    if levels == 0:
        return z

    n = z.shape[0]//(2**levels)
    m = 2*n
    cA = z[:n,:n]
    cH = z[:n,n:m]
    cV = z[n:m,:n]
    cD = z[n:m,n:m]

    z = z.copy()
    z[:m, :m] = pywt.idwt2((cA, (cV, cH, cD)), wavelet, mode)

    return idwt2(z, levels-1)
