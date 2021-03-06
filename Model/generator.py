from tensorflow.keras.layers import Conv2D, Activation, Input, Reshape, BatchNormalization
from Model.ops import up_res_block, dense_spectral_norm, conv_spectral_norm
from Model.layers import SelfAttention
from tensorflow.keras.models import Model

def make_sagan_generator_model(img_dim, noise_shape, initial_filters, gen_kernel_size, kernel_init):

  num_filters = initial_filters

  gen_input = Input(shape = noise_shape)

  generator = dense_spectral_norm(gen_input,4*4*num_filters,True)
  
  generator = Reshape((4,4,num_filters))(generator)
  
  generator = up_res_block(generator,num_filters, gen_kernel_size, kernel_init)
  
  generator = up_res_block(generator,num_filters // 2, gen_kernel_size, kernel_init)
  
  generator = up_res_block(generator,num_filters // 4, gen_kernel_size, kernel_init)

  generator = SelfAttention()(generator)

  generator = up_res_block(generator,num_filters // 8, gen_kernel_size, kernel_init)

  generator = up_res_block(generator,num_filters // 16, gen_kernel_size, kernel_init)

  generator = BatchNormalization()(generator)

  generator = Activation('relu')(generator)
  
  generator = conv_spectral_norm(generator, 3, gen_kernel_size, 1, kernel_init, True) 

  generator = Activation('tanh')(generator)
  
  generator_model = Model( gen_input, generator)

  return generator_model
  