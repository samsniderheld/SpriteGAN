from tensorflow.keras.layers import Dense, Input, LeakyReLU
from Model.ops import down_res_block, dense_spectral_norm
from Model.layers import SelfAttention
from tensorflow.keras.models import Model
import tensorflow as tf

def make_sagan_discriminator_model(img_dim, disc_kernel_size, kernel_init):
    
  image_shape = (img_dim,img_dim,3)
  
  dis_input = Input(shape = image_shape)

  num_filters = 64

  discriminator = down_res_block(dis_input,num_filters, disc_kernel_size, kernel_init)
  
  discriminator = down_res_block(discriminator,num_filters * 2, disc_kernel_size, kernel_init)

  discriminator = SelfAttention()(discriminator)
  
  discriminator = down_res_block(discriminator,num_filters * 4, disc_kernel_size, kernel_init)
  
  discriminator = down_res_block(discriminator,num_filters * 8, disc_kernel_size, kernel_init)

  discriminator = down_res_block(discriminator,num_filters * 16, disc_kernel_size, kernel_init)

  discriminator = down_res_block(discriminator,num_filters * 32, disc_kernel_size, kernel_init)
  
  discriminator = LeakyReLU(.2)(discriminator)

  discriminator = tf.reduce_sum(discriminator, axis=[1, 2])
  
  discriminator = Dense(1)(discriminator)

  discriminator  = dense_spectral_norm(discriminator,1,True)
    
  discriminator_model = Model(dis_input, discriminator)  
  
  return discriminator_model