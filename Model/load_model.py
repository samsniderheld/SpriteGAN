from Model.discriminator import make_sagan_discriminator_model
from Model.generator import make_sagan_generator_model
import tensorflow as tf

def load_models_from_step(args):
	kernel_init = tf.keras.initializers.GlorotNormal()
	noise_shape = (1,1,args.noise_dim)
	discriminator = make_sagan_discriminator_model(args.img_dim, args.disc_kernel_size, kernel_init)
	generator = make_sagan_generator_model(args.img_dim, noise_shape, args.initial_gen_filters, args.gen_kernel_size, kernel_init)

	discriminator.load_weights("SavedModels/discriminator_weights_at_step_{}.h5".format(args.step))
	generator.load_weights("SavedModels/generator_weights_at_step_{}.h5".format(args.step))

	return discriminator, generator


def load_generator_from_step(args):
	kernel_init = tf.keras.initializers.GlorotNormal()
	noise_shape = (1,1,args.noise_dim)
	
	generator = make_sagan_generator_model(args.img_dim, noise_shape, args.initial_gen_filters, args.gen_kernel_size, kernel_init)

	generator.load_weights("SavedModels/generator_weights_at_step_{}.h5".format(args.step))

	return generator