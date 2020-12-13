from tensorflow.keras.layers import Dense, Input, LeakyReLU
from Model.ops import down_res_block, dense_spectral_norm, down_res_block_2, down_res_block_2_init, final_block, attention_3
# from Model.layers import SelfAttention, SelfAttention2
from tensorflow.keras.models import Model
import tensorflow as tf

def make_sagan_discriminator_model(img_dim, disc_kernel_size, kernel_init):
    
  image_shape = (img_dim,img_dim,3)
  
  dis_input = Input(shape = image_shape)

  num_filters = 64

  discriminator = down_res_block_2_init(dis_input,num_filters, disc_kernel_size, kernel_init)
  
  discriminator = down_res_block_2(discriminator,num_filters * 2, disc_kernel_size, kernel_init)

  # discriminator = down_res_block(dis_input,num_filters, disc_kernel_size, kernel_init)
  
  # discriminator = down_res_block(discriminator,num_filters * 2, disc_kernel_size, kernel_init)

  discriminator = attention_3(discriminator, num_filters * 2)

  # discriminator = down_res_block(discriminator,num_filters * 4, disc_kernel_size, kernel_init)
  
  # discriminator = down_res_block(discriminator,num_filters * 8, disc_kernel_size, kernel_init)

  # discriminator = down_res_block(discriminator,num_filters * 16, disc_kernel_size, kernel_init)
  
  discriminator = down_res_block_2(discriminator,num_filters * 4, disc_kernel_size, kernel_init)
  
  discriminator = down_res_block_2(discriminator,num_filters * 8, disc_kernel_size, kernel_init)

  discriminator = down_res_block_2(discriminator,num_filters * 16, disc_kernel_size, kernel_init)

  discriminator = final_block(discriminator,num_filters * 16, disc_kernel_size, kernel_init)
  
  discriminator = LeakyReLU(.2)(discriminator)

  discriminator = tf.reduce_sum(discriminator, axis=[1, 2])
  
  discriminator = dense_spectral_norm(discriminator,1,True)
    
  discriminator_model = Model(dis_input, discriminator)  
  
  return discriminator_model