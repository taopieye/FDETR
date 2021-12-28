import tensorflow as tf
from tensorflow.keras.layers import Reshape, \
                                    Dropout, \
                                    Add, \
                                    Normalization, \
                                    MultiHeadAttention, \
                                    Dense

from FftAttn import FftAttn

from unused import MinMax


class Transformer(tf.keras.layers.Layer):
    def __init__(self, batch_size, d_model=256, dff=2048, dropout=0.1, num_heads=4, 
                 num_encoder_layers=2, num_decoder_layers=2):
        super(Transformer, self).__init__()

        self.batch_size = batch_size

        self.encoder = TransformerEncoder(d_model=d_model, dff=dff, dropout=dropout, num_layers=num_encoder_layers)

        self.decoder = TransformerDecoder(d_model=d_model, dff=dff, dropout=dropout, num_heads=num_heads, num_layers=num_decoder_layers)

    def call(self, inputs, query_embed, pos_enc, training=None):
        _, x, y, channels = inputs.shape
        inputs = Reshape((x * y, channels))(inputs)
        pos_enc = Reshape((x * y, channels))(pos_enc)
        query_embed = tf.expand_dims(query_embed, 0)
        query_embed = tf.tile(query_embed, [self.batch_size, 1, 1])

        memory = self.encoder(inputs, pos_enc, training=training)
        out = self.decoder(query_embed, memory, pos_enc, training=training)

        return out

class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, d_model=256, 
                 dff=2048, dropout=0.0, num_layers=2):
        super(TransformerEncoder, self).__init__()

        self.d_model = d_model

        self.encoder_layers = [EncoderLayer(d_model=d_model, dff=dff, dropout=dropout) for _ in range(num_layers)] 

        self.dropout = Dropout(dropout)

    def call(self, inputs, pos_enc, training=None):
        inputs += pos_enc
        
        x = self.dropout(inputs, training=training)

        #encoder layer
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, training=training)
        
        return x

class TransformerDecoder(tf.keras.layers.Layer):
    def __init__(self, d_model=256,  
                 dff=2048, dropout=0.0,
                 num_heads=4, num_layers=2):
        super(TransformerDecoder, self).__init__()

        self.d_model = d_model

        self.layers = [DecoderLayer(d_model=d_model, num_heads=num_heads, dff=dff, dropout=dropout) for _ in range(num_layers)]

    def call(self, inputs, memory, pos_enc, training=None):
        output = inputs

        for layer in self.layers:
            output = layer(output, memory, pos_enc, training=training)
        
        return output

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model=256, dff=2048, dropout=0.0):
        super(EncoderLayer, self).__init__()

        self.self_attn = FftAttn()
        self.add_attn = Add()
        self.norm_attn = Normalization()

        self.dense1 = Dense(dff, activation=tf.nn.gelu)
        self.dense2 = Dense(d_model)
        self.dropout_dense = Dropout(dropout)
        self.add_dense = Add()
        self.norm_dense = Normalization()

    def call(self, inputs, mask=None, training=None):
        attn = self.self_attn(inputs)
        x = self.add_attn([inputs, attn])
        x = self.norm_attn(x)

        #feed that shit forward
        dense = self.dense1(x)
        dense = self.dense2(dense)
        dense = self.dropout_dense(dense, training=training)
        x = self.add_dense([x, dense])
        x = self.norm_dense(x)

        return x

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model=256, dff=2048, dropout=0.0, num_heads=4):
        super(DecoderLayer, self).__init__()

        self.self_attn = FftAttn()
        self.dropout_attn1 = Dropout(dropout)
        self.add_attn1 = Add()
        self.norm_attn1 = Normalization()

        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=d_model, dropout=dropout)
        self.dropout_mha = Dropout(dropout)
        self.add_mha = Add()
        self.norm_mha = Normalization()

        self.dense1 = Dense(dff, activation=tf.nn.gelu)
        self.dense2 = Dense(d_model)
        self.dropout_dense = Dropout(dropout)
        self.add_dense = Add()
        self.norm_dense = Normalization()

        self.d_model = d_model

    def call(self, inputs, memory, pos_enc, training=None):
        x_in = inputs
        fft_attn = self.self_attn(x_in)
        fft_attn = self.dropout_attn1(fft_attn, training=training)
        
        x = self.add_attn1([x_in, fft_attn])
        x = self.norm_attn1(x)

        q = Dense(self.d_model)(Add()([x, x_in]))
        k = Dense(self.d_model)(Add()([memory, pos_enc]))
        v = Dense(self.d_model)(memory)

        #multihead cross-attention
        mha = self.mha(query=q, key=k, value=v, return_attention_scores=False)

        x = self.add_mha([x, mha])
        x = self.norm_mha(x)

        #feed that shit forward
        dense = self.dense1(x)
        dense = self.dense2(dense)
        dense = self.dropout_dense(dense, training=training)
        x = self.add_dense([x, dense])
        x = self.norm_dense(x)

        return x