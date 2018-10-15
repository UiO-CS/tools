import numpy as np

from .pywtwrappers import dwt2, idwt2, dwt, idwt

# TODO: Better implemented as a class?
def fourier_wavelet_2d(wavelet, levels, mask):
    """Creates functions that calculate the forward and backward transforms

    Arguments:
        wavelet (string): Name of wavelet to use
        levels (Int): Number of layers for dwt to perform.
        mask (ndarray): Boolean mask of same dimensions as input to the
                        transform. The values corresponding to True will be kept.

        Note that fftshift is not performed. The mask should therefore be
        reordered appropriately.

    Returns:
        Two functions that take an ndarray, and computes the forward an backwards transform
    """

    def forward(x):
        """P_\Omega F W*

        Where P_\Omega(x)_j is x_j if j in \Omega, else 0. P : C^N -> C^N

        Note that fftshift is not performed. The mask should therefore be
        reordered appropriately.
        """
        result = idwt2(x, wavelet, levels)
        result = np.fft.fft2(result)
        result[~mask] = 0
        return result


    def adjoint(x):
        """W F*

        Adjoint of P_\Omega is not necessary, as the forward transform actually
        calculates P_\Omega* P_\Omega

        """
        result = np.fft.ifft2(x)
        result = dwt2(result, wavelet, levels)

        return result


    return forward, adjoint


def fourier_wavelet_1d(wavelet, levels, mask):
    """See fourier_wavelet_2d

    Written primarily to confirm that the eigenvalue is 1 by power iterations.
    It seems that the mask makes the system too unstable"""

    def forward(x):
        result = idwt(x, wavelet, levels)
        result = 1./np.sqrt(x.size) * np.fft.fft(result)
        result[~mask] = 0
        return result


    def adjoint(x):
        result = np.fft.ifft(np.sqrt(x.size) * x)
        result = dwt(result, wavelet, levels)

        return result

    return forward, adjoint
