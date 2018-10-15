import tensorflow as tf


def convert_complex_format(tensor_in):
    """
    Converts complex number format from using the actual tf.complex64 type (used
    for tf.spectral.fft) and having the real and imaginary parts as separate
    channels in the tensor (used for everything else).

    Args:
        tensor_in:      A 4D tensor [batch, height, width, channels] where either:

                          * The channels dimension is 1, and the type is complex

                                or

                          * The channels dimemsion is 2, and the type is some
                            type of real number (float, int) 

    Returns:
        The tensor where the complex type has been toggled.
    """
    if tensor_in.dtype.is_complex:
        # Convert to 2 channels
        return tf.concat([tf.real(tensor_in), tf.imag(tensor_in)], axis=3)

    else:
        # Convert to complex type
        return tf.complex(real=tensor_in[:, :, :, 0:1], imag=tensor_in[:, :, :, 1:])
