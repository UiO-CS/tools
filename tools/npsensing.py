import numpy as np

from .pywtwrappers import dwt2, idwt2

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
        result = np.fft.ifft2(result)
        result[~mask] = 0
        return result


    def adjoint(x):
        """W F*

        Adjoint of P_\Omega is not necessary, as the forward transform actually
        calculates P_\Omega* P_\Omega

        """
        result = np.fft.ifft2(result)
        result = dwt2(result, wavelet, levels)


    return forward, adjoint


