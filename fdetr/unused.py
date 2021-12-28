import tensorflow as tf
from tensorflow.keras.layers import Dense

class MinMax(tf.keras.layers.Layer):
    def __init__(self, t_min=0.0, t_max=1.0):
        super(MinMax, self).__init__()
        self.t_min = t_min
        self.t_rng = t_max - t_min

    def call(self, inputs):
        x_min = tf.math.reduce_min(inputs)
        x_max = tf.math.reduce_max(inputs)
        rng = x_max - x_min
        minmax = (inputs - x_min) / rng
        return minmax * self.t_rng + self.t_rng 

class MLP(tf.keras.layers.Layer):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = [Dense(k, activation='relu') for k in h] + [Dense(output_dim, activation='sigmoid')]

    def call(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)

        return x 