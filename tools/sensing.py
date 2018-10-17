import numpy as np
import tensorflow as tf

from tfwavelets.dwtcoeffs import db2
from tfwavelets.nodes import dwt2d, idwt2d



def fourier_wavelet_2d(wavelet, levels, mask, nsqrt):
    '''
    Tensorflow equivalent to the function with the same name in npsensing.

    Arguments:
        wavelet: A wavelet object
        levels: Number of levels in DWT
        mask: boolean Tensor of same shape as input to resulting operations.
              Values where mask is true will be kept
        nsqrt: FFT will be calculated as `1/nsqrt * FFT`, and ifft will be
               calculated `nsqrt * IFFT`, to make fourier transform unitary.

    Returns:
        Callable forward and adjoint transforms.

        `forward` calculates P FW* where P is the projection matrix that keeps
        values accoring to mask, and zeros out the rest, F is the Fourier tranform,
        and W the discrete wavelet transform

        `adjoint` calculates the adjoint i.e W F* P*
    '''


    def forward(x):

        # Compute the IDWT both for real and imaginary part
        # tf.conv1d does not support complex numbers
        # tfWavelets only support 3D-tensors
        real_idwt = idwt2d(tf.real(x), wavelet, levels)
        imag_idwt = idwt2d(tf.imag(x), wavelet, levels)
        complex_idwt = tf.complex(real_idwt, imag_idwt)

        # FFT2 uses the two last dimensions
        result = tf.transpose(complex_idwt, [2,0,1]) # [channels, height, width]
        # TODO: Scaling?
        result = 1./nsqrt * tf.fft2d(result)
        result = tf.transpose(result, [1,2,0]) # [height, width, channels]

        result = tf.where(mask, result, tf.zeros_like(result))
        return result

    def adjoint(x):
        result = tf.where(mask, x, tf.zeros_like(x))

        # Calculate IFFT
        result = tf.transpose(result, [2,0,1]) # [channels, height, width]
        result = nsqrt * tf.ifft2d(result)
        result = tf.transpose(result, [1,2,0]) # [height, width, channels]

        real_dwt = dwt2d(tf.real(result), wavelet, levels)
        imag_dwt = dwt2d(tf.imag(result), wavelet, levels)
        result = tf.complex(real_dwt, imag_dwt)
        return result

    return forward, adjoint

