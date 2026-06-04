import tensorflow as tf
from tensorflow.keras import layers, Model



# Binary Linear Layer
class BinaryLinear(layers.Layer):
    def __init__(self, out_features):
        super(BinaryLinear, self).__init__()
        self.out_features = out_features

    def build(self, input_shape):
        in_features = input_shape[-1]

        # Initialize small random weights
        self.w = self.add_weight(
            shape=(in_features, self.out_features),
            initializer=tf.keras.initializers.RandomUniform(
                minval=-0.001,
                maxval=0.001
            ),
            trainable=True,
            name='binary_weight'
        )

    def call(self, x):

        # Real weights
        real_weights = self.w

        # Scaling factor
        scaling_factor = tf.stop_gradient(
            tf.reduce_mean(tf.abs(real_weights))
        )

        # Binary weights
        binary_weights_no_grad = (
            scaling_factor * tf.sign(real_weights)
        )

        # Clipping
        clipped_weights = tf.clip_by_value(
            real_weights,
            -1.0,
            1.0
        )

        # Straight-Through Estimator (STE)
        binary_weights = (
            tf.stop_gradient(
                binary_weights_no_grad - clipped_weights
            )
            + clipped_weights
        )

        # Linear transformation
        y = tf.matmul(x, binary_weights)

        return y



# BHDC Model


class BHDC(Model):
    def __init__(self,
                 inshape=10000,
                 outshape=10,
                 dropout_prob=0.0):

        super(BHDC, self).__init__()

        self.inshape = inshape

        self.dropout = layers.Dropout(dropout_prob)

        self.binary_layer = BinaryLinear(outshape)

    def call(self, x, training=False):

        # Flatten
        x = tf.reshape(x, [-1, self.inshape])

        # Dropout
        x = self.dropout(x, training=training)

        # Binary linear layer
        x = self.binary_layer(x)

        return x