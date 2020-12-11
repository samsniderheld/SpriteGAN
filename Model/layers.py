import tensorflow as tf
from ops import *
from tensorflow.keras.layers import MaxPool2D


#via https://github.com/grohith327/simplegan/blob/master/simplegan/layers/spectralnorm.py
class SpectralNormalization(tf.keras.layers.Wrapper):
    """
    Paper: https://arxiv.org/pdf/1802.05957.pdf
    Implementation based on https://github.com/tensorflow/addons/pull/1244

    Spectral norm is computed using power iterations.

    Attributes:
        layer (tf.keras.layer): Input layer to be taken spectral norm
        power_iternations (int): Number of power iterations to approximate the values"
    """

    def __init__(self, layer, power_iterations=1, **kwargs):
        super(SpectralNormalization, self).__init__(layer, **kwargs)
        if power_iterations <= 0:
            raise ValueError(
                "`power_iterations` should be greater than zero, got "
                "`power_iterations={}`".format(power_iterations)
            )
        self.power_iterations = power_iterations


    def build(self, input_shape):

        input_shape = tf.TensorShape(input_shape).as_list()
        self.layer.build(input_shape)

        if hasattr(self.layer, "kernel"):
            self.w = self.layer.kernel
        elif hasattr(self.layer, "embeddings"):
            self.w = self.layer.embeddings
        else:
            raise AttributeError(
                "{} object has no attribute 'kernel' nor "
                "'embeddings'".format(type(self.layer).__name__)
            )

        self.w_shape = self.w.shape.as_list()

        self.u = self.add_weight(
            shape=(1, self.w_shape[-1]),
            initializer=tf.initializers.TruncatedNormal(stddev=0.02),
            trainable=False,
            name="sn_u",
            dtype=self.layer.kernel.dtype,
        )

        super(SpectralNormalization, self).build()

    def call(self, inputs):
        self.normalize_weights()
        output = self.layer(inputs)
        return output

    def normalize_weights(self):
        w = tf.reshape(self.w, [-1, self.w_shape[-1]])
        u = self.u

        for i in range(self.power_iterations):
            v = tf.math.l2_normalize(tf.matmul(u, tf.transpose(w)))
            u = tf.math.l2_normalize(tf.matmul(v, w))

        sigma = tf.matmul(tf.matmul(v, w), tf.transpose(u))

        self.w.assign(self.w / sigma)
        self.u.assign(u)


#via https://github.com/grohith327/simplegan/blob/master/simplegan/layers/selfattention.py
class SelfAttention(tf.keras.Model):
    def __init__(self, spectral_norm=True):
        super(SelfAttention, self).__init__()
        self.scaling_factor = tf.Variable(0.0)
        self.spectral_norm = spectral_norm

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'scaling_factor': self.scaling_factor,
            'spectral_norm': self.spectral_norm,
        })
        return config


    def build(self, input):
        _, _, _, n_channels = input

        if self.spectral_norm:
            self.conv1x1_f = SpectralNormalization(
                tf.keras.layers.Conv2D(
                    filters=n_channels // 8, kernel_size=(1, 1), padding="same", strides=(1, 1)
                )
            )
            self.conv1x1_g = SpectralNormalization(
                tf.keras.layers.Conv2D(
                    filters=n_channels // 8, kernel_size=(1, 1), padding="same", strides=(1, 1)
                )
            )
            self.conv1x1_h = SpectralNormalization(
                tf.keras.layers.Conv2D(
                    filters=n_channels // 2, kernel_size=(1, 1), padding="same", strides=(1, 1)
                )
            )

            self.conv1x1_attn = SpectralNormalization(
                tf.keras.layers.Conv2D(
                    filters=n_channels, kernel_size=(1, 1), padding="same", strides=(1, 1)
                )
            )
        else:
            self.conv1x1_f = tf.keras.layers.Conv2D(
                filters=n_channels // 8, kernel_size=(1, 1), padding="same", strides=(1, 1)
            )
            self.conv1x1_g = SpectralNormalization(
                tf.keras.layers.Conv2D(
                    filters=n_channels // 8, kernel_size=(1, 1), padding="same", strides=(1, 1)
                )
            )
            self.conv1x1_h = SpectralNormalization(
                tf.keras.layers.Conv2D(
                    filters=n_channels // 2, kernel_size=(1, 1), padding="same", strides=(1, 1)
                )
            )

            self.conv1x1_attn = SpectralNormalization(
                tf.keras.layers.Conv2D(
                    filters=n_channels, kernel_size=(1, 1), padding="same", strides=(1, 1)
                )
            )

        self.f_maxpool = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)
        self.g_maxpool = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)
        self.h_maxpool = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)


    def call(self, x):
        batch_size, height, width, n_channels = x.shape
        f = self.conv1x1_f(x)
        # f = self.f_maxpool(f)
        # f = tf.reshape(f, [tf.shape(x)[0], (height * width) // 4, n_channels // 8])
        f = tf.reshape(f, [tf.shape(x)[0], height * width, n_channels // 8])

        g = self.conv1x1_g(x)
        g = self.g_maxpool(g)
        g = tf.reshape(g, [tf.shape(x)[0], (height * width) // 4, n_channels // 8])
        # g = tf.reshape(g, [tf.shape(x)[0], height * width, n_channels // 8])

        attn_map = tf.matmul(f, g, transpose_b=True)
        attn_map = tf.nn.softmax(attn_map)

        h = self.conv1x1_h(x)
        h = self.h_maxpool(h)
        h = tf.reshape(h, [tf.shape(x)[0], (height * width) // 4, n_channels // 2])

        attn_h = tf.matmul(attn_map, h)
        attn_h = tf.reshape(attn_h, [tf.shape(x)[0], width, height, n_channels // 2])
        attn_h = self.conv1x1_attn(attn_h)

        out = x + (attn_h * self.scaling_factor)

        return out


class SelfAttention2(tf.keras.Model):
    def __init__(self, spectral_norm=True):
        super(SelfAttention, self).__init__()
        self.scaling_factor = tf.Variable(0.0)
        self.spectral_norm = spectral_norm

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'scaling_factor': self.scaling_factor,
            'spectral_norm': self.spectral_norm,
        })
        return config


    def build(self, input):
        


    def call(self, x):
        kernel_init = tf.keras.initializers.GlorotUniform()
        batch_size, height, width, num_channels = x.get_shape().as_list()
        f = conv_spectral_norm(x, num_channels // 8, 1, 1,kernel_init,True) # [bs, h, w, c']
        f = MaxPool2D()(f)

        g = conv_spectral_norm(x, num_channels // 8, 1, 1,kernel_init,True) # [bs, h, w, c']

        h = conv_spectral_norm(x, num_channels // 2, 1, 1,kernel_init,True) # [bs, h, w, c']
        h = MaxPool2D()(h)

        # N = h * w
        s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)  # # [bs, N, N]

        beta = tf.nn.softmax(s)  # attention map

        o = tf.matmul(beta, hw_flatten(h))  # [bs, N, C]
        gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))

        o = tf.reshape(o, shape=[batch_size, height, width, num_channels // 2])  # [bs, h, w, C]
        o = conv_spectral_norm(o, num_channels , 1, 1,kernel_init,True) # [bs, h, w, c']
        x = gamma * o + x

        return x