import tensorflow as tf
import numpy as np

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Input, Conv2D

def cnn_backbone(d_model, batch_size, input_shape):

    resnet_input = Input(shape=input_shape)

    resnet = ResNet50(
        include_top=False, 
        input_tensor=resnet_input, 
        weights='imagenet',
    )

    for layer in resnet.layers:
        layer.trainable = True

    c3_output = resnet.get_layer('conv3_block4_out').output

    resnet_features = Conv2D(d_model, 1, 2)(c3_output)

    pos_enc = PosEncoding2D(batch_size)(resnet_features)

    return tf.keras.Model(inputs=resnet_input, outputs=[resnet_features, pos_enc])

class PosEncoding2D(tf.keras.layers.Layer):
    def __init__(self, batch_size):
        super(PosEncoding2D, self).__init__()
        self.batch_size = batch_size
    
    def call(self, inputs):
        '''
        :inputs: 4d tensor of size (batch_size, x, y, channels)
        :return: positional encoding matrix of same size as inputs
        '''
        if len(inputs.shape) != 4:
            raise RuntimeError("Inputs must be 4d")
        
        _, x, y, og_chan = inputs.shape

        channels = int(np.ceil(og_chan/4)*2)
        inv_freq = 1. / (10000 ** (np.arange(0, channels, 2).astype(np.float32) / channels))
        
        #get pixel positions
        pos_x = np.arange(x).astype(np.float32)
        pos_y = np.arange(y).astype(np.float32)
        
        #get sin/cos inputs
        sin_inp_x = np.einsum('i,j->ij', pos_x, inv_freq)
        sin_inp_y = np.einsum('i,j->ij', pos_y, inv_freq)
        
        enc_x = np.concatenate((np.sin(sin_inp_x), np.cos(sin_inp_x)), axis=-1)
        enc_y = np.concatenate((np.sin(sin_inp_y), np.cos(sin_inp_y)), axis=-1)
        
        enc = np.zeros((x, y, channels*2)).astype(np.float32)
        
        #populate embedding matrix
        enc[:, :, :channels] = enc_x
        enc[:, :, channels:channels*2] = enc_y

        #make tf tensor
        enc = tf.convert_to_tensor(enc[:, :, :og_chan], dtype=tf.float32)
        
        enc = tf.expand_dims(enc, 0)

        enc = tf.tile(enc, [self.batch_size, 1, 1, 1])
        
        return enc