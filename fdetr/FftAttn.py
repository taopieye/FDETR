import tensorflow as tf
from tensorflow.keras.layers import Permute


class FftAttn(tf.keras.layers.Layer):
    '''Mixes tokens using Fast Fourier Transform
    
    Arguments: N/A

    Returns:
        The Real part of the 2D fft of a tensor of shape
        (batch_size, seq_len, hidden_dim), calculated
        by taking the fft along the hidden dim, then
        along the seq_len dim
    
    '''
    def __init__(self):
        super(FftAttn, self).__init__()

    def call(self, inputs):
        inputs_complex = tf.cast(inputs, tf.complex64)
        #get fft along hidden dim
        f_h = tf.signal.fft(inputs_complex)
        #now get fft along seq dim
        x = Permute((2,1))(f_h)
        f_seq = tf.signal.fft(x)
        #recover initial shape
        x = Permute((2,1))(f_seq)
        #take real part of transform
        x = tf.math.real(x)
        return tf.cast(x, tf.float32)