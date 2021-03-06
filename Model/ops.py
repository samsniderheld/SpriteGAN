from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, LeakyReLU, Add, AveragePooling2D, ReLU, MaxPool2D
from Model.layers import SpectralNormalization
import tensorflow as tf


#via https://github.com/manicman1999/Keras-BiGAN/blob/master/bigan.py
def up_res_block(input, filters, gen_kernel_size, kernel_init):

  skip = up_sample(input)
  skip = conv_spectral_norm(skip, filters, gen_kernel_size, 1, kernel_init, False, pad_type='zero')

  output = BatchNormalization()(input)
  output = ReLU()(output)
  output = up_sample(output)
  output = conv_spectral_norm(output,filters, gen_kernel_size, 1, kernel_init, False)
  
  output = BatchNormalization()(output)
  output = ReLU()(output)
  output = conv_spectral_norm(output, filters,gen_kernel_size,1,kernel_init, True)

  output = Add()([output, skip])

  return output

#via https://github.com/taki0112/Self-Attention-GAN-Tensorflow/blob/master/ops.py
def down_res_block(input, filters, disc_kernel_size, kernel_init):

  skip = conv_spectral_norm(input, filters, disc_kernel_size, 1,kernel_init,True, pad_type='zero')
  skip = AveragePooling2D()(skip)

  output = LeakyReLU(0.2)(input)
  output = conv_spectral_norm(output, filters, disc_kernel_size, 1, kernel_init, True)
  
  output = LeakyReLU(0.2)(output)
  output = conv_spectral_norm(output, filters, disc_kernel_size, 1, kernel_init, True) 

  output = AveragePooling2D()(output)

  output = Add()([output, skip])

  return output

def dense_spectral_norm(input,filters,bias):

  spectralDense = SpectralNormalization(
      Dense(filters,use_bias=bias)
    )

  return spectralDense(input)

def conv_spectral_norm(input, filters, kernel_size, stride, kernel_init, bias, pad_type='reflect'):

  spectralConv = SpectralNormalization(
    Conv2D(filters, kernel_size = (kernel_size,kernel_size), strides = (stride,stride), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init, use_bias=bias)
  )

  return spectralConv(input)

def up_sample(input):
  _, h, w, _ = input.get_shape().as_list()
  new_size = [h * 2, w * 2]
  return tf.image.resize(input, size=new_size, method='nearest')

def down_sample(input):
  return AveragePooling2D(input, pool_size=(2,2), strides = (2,2))

def hw_flatten(x) :
    return tf.reshape(x, shape=[tf.shape(x)[0], -1, tf.shape(x)[-1]])


