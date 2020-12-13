from tensorflow.keras.layers import Conv2D, Activation, Input, Reshape, BatchNormalization
from Model.ops import up_res_block, dense_spectral_norm, conv_spectral_norm, attention_3
# from Model.layers import SelfAttention, SelfAttention2
from tensorflow.keras.models import Model


def make_sagan_generator_model(img_dim, noise_shape, gen_kernel_size, kernel_init):

  num_filters = 1024

  gen_input = Input(shape = noise_shape)

  generator = dense_spectral_norm(gen_input,4*4*num_filters,True)
  
  generator = Reshape((4,4,num_filters))(generator)
  
  generator = up_res_block(generator,num_filters, gen_kernel_size, kernel_init)
  
  generator = up_res_block(generator,num_filters // 2, gen_kernel_size, kernel_init)
  
  generator = up_res_block(generator,num_filters // 4, gen_kernel_size, kernel_init)

  generator = attention_3(generator, num_filters // 4)

  generator = up_res_block(generator,num_filters // 8, gen_kernel_size, kernel_init)

  generator = up_res_block(generator,num_filters // 16, gen_kernel_size, kernel_init)

  generator = BatchNormalization()(generator)

  generator = Activation('relu')(generator)

  # generator = Conv2D(filters = 3, kernel_size = (gen_kernel_size,gen_kernel_size), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init, use_bias=True)(generator)
  
  generator = conv_spectral_norm(generator, 3, gen_kernel_size, 1, kernel_init, True) 

  generator = Activation('tanh')(generator)
  
  generator_model = Model( gen_input, generator)

  return generator_model
  