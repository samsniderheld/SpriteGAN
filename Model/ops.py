from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, LeakyReLU, Add, AveragePooling2D, ReLU
from Model.layers import SpectralNormalization
import tensorflow as tf


#via https://github.com/manicman1999/Keras-BiGAN/blob/master/bigan.py
def up_res_block(input, filters, gen_kernel_size, kernel_init):

  skip = up_sample(input)
  skip = conv_spectral_norm(skip, filters, 1, 1, kernel_init, False)

  output = BatchNormalization()(input)
  output = ReLU()(output)
  output = up_sample(output)
  output = conv_spectral_norm(output,filters, gen_kernel_size, 1, kernel_init, False)
  
  output = BatchNormalization()(output)
  output = ReLU()(output)
  output = conv_spectral_norm(output, filters,gen_kernel_size,1,kernel_init, True)

  output = Add()([output, skip])

  return output

#via https://github.com/manicman1999/Keras-BiGAN/blob/master/bigan.py
def down_res_block(input, filters, disc_kernel_size, kernel_init):

  skip = conv_spectral_norm(input, filters, disc_kernel_size, 1,kernel_init,True)

  output = conv_spectral_norm(input, filters, disc_kernel_size, 1, kernel_init, True)
  output = LeakyReLU(0.2)(output)

  output = conv_spectral_norm(output, filters, disc_kernel_size, 1, kernel_init, True)
  output = LeakyReLU(0.2)(output)

  output = conv_spectral_norm(output, filters, disc_kernel_size, 1, kernel_init, True)

  output = Add()([output, skip])
  output = LeakyReLU(0.2)(output)

  output = AveragePooling2D()(output)


  return output

#via https://github.com/taki0112/Self-Attention-GAN-Tensorflow/blob/master/ops.py
def down_res_block_2(input, filters, disc_kernel_size, kernel_init):

  skip = conv_spectral_norm(input, filters, disc_kernel_size, 1,kernel_init,True)
  skip = AveragePooling2D()(skip)

  output = LeakyReLU(0.2)(input)
  output = conv_spectral_norm(output, filters, disc_kernel_size, 1, kernel_init, True)
  
  output = LeakyReLU(0.2)(output)
  output = conv_spectral_norm(output, filters, disc_kernel_size, 1, kernel_init, True) 

  output = AveragePooling2D()(output)

  output = Add()([output, skip])

  return output


def final_block(input, filters, disc_kernel_size, kernel_init):

  # skip = conv_spectral_norm(input, filters, disc_kernel_size, 1,kernel_init,True)

  output = LeakyReLU(0.2)(input)
  output = conv_spectral_norm(output, filters, disc_kernel_size, 1, kernel_init, True)
  
  output = LeakyReLU(0.2)(output)
  output = conv_spectral_norm(output, filters, disc_kernel_size, 1, kernel_init, True) 

  # output = Add()([skip, output])

  return output

def down_res_block_2_init(input, filters, disc_kernel_size, kernel_init):

  skip = AveragePooling2D()(input)
  skip = conv_spectral_norm(skip, filters, disc_kernel_size, 1,kernel_init,True)

  output = conv_spectral_norm(input, filters, disc_kernel_size, 1, kernel_init, True)
  output = LeakyReLU(0.2)(output)

  output = conv_spectral_norm(output, filters, disc_kernel_size, 1, kernel_init, True)
  output = AveragePooling2D()(output)

  output = Add()([output, skip])


  return output



def dense_spectral_norm(input,filters,bias):

  # spectralDense = SpectralNormalization(
  #     Dense(filters,use_bias=bias)
  #   )

  # return spectralDense(input)

  weight_init = tf.keras.initializers.GlorotUniform()

  x = tf.layers.flatten(input)
  shape = x.get_shape().as_list()
  channels = shape[-1]

  # w = tf.get_variable("kernel", [channels, filters], tf.float32, initializer=weight_init, regularizer=None)
  w = tf.Variable(shape=(channels, filters))

  bias = tf.get_variable(0, shape = (filters))

  x = tf.matmul(x, spectral_norm(w)) + bias


  return x

def conv_spectral_norm(input, filters, kernel_size, stride, kernel_init, bias):

  # spectralConv = SpectralNormalization(
  #   Conv2D(filters, kernel_size = (kernel_size,kernel_size), strides = (stride,stride), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init, use_bias=bias)
  # )

  # return spectralConv(input)

  h = input.get_shape().as_list()[1]

  pad = 1

  if h % stride == 0:
      pad = pad * 2
  else:
      pad = max(kernel_size - (h % stride), 0)

  pad_top = pad // 2
  pad_bottom = pad - pad_top
  pad_left = pad // 2
  pad_right = pad - pad_left

  x = tf.pad(input, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], mode='REFLECT')

  # w = tf.get_variable("kernel", shape=[kernel_size, kernel_size, x.get_shape()[-1],filters], initializer=kernal_init)
  w = tf.Variable(shape=(kernel_size, kernel_size, x.get_shape()[-1],filters))

  x = tf.nn.conv2d(input=x, filter=spectral_norm(w), strides=[1, stride, stride, 1], padding='VALID',, initializer=kernal_init)

  # bias = tf.get_variable("bias", [filters], initializer=tf.constant_initializer(0.0))

  bias = tf.Variable(0, shape = (filters))

  x = tf.nn.bias_add(x, bias)

  return x

def up_sample(input):
  _, h, w, _ = input.get_shape().as_list()
  new_size = [h * 2, w * 2]
  return tf.image.resize(input, size=new_size, method='nearest')

def down_sample(input):
  return AveragePooling2D(input, pool_size=(2,2), strides = (2,2))

def spectral_norm(w, iteration=1):
    w_shape = w.get_shape().as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    # u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)
    u = tf.Variable(shape=(1, w_shape[-1]))

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """

        # v = tf.math.l2_normalize(tf.matmul(u, tf.transpose(w)))
        # u = tf.math.l2_normalize(tf.matmul(v, w))

        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.math.l2_normalize(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = tf.math.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm
