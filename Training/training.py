import tensorflow as tf
from Model.discriminator import make_sagan_discriminator_model
from Model.generator import make_sagan_generator_model
from Model.load_model import load_models_from_step
from Training.loss_functions import sagan_discriminator_loss, sagan_generator_loss
from Utils.reporting import generate_and_save_images, plot_loss
import time
import glob

@tf.function
def train_step(discriminator, generator, d_op, g_op, images, noise):

  #train and upate discriminator on real data
  with tf.GradientTape() as disc_tape:
    real_output = discriminator(images, training=True)
    generated_images = generator(noise, training=True)
    fake_output = discriminator(generated_images, training=True)

    disc_loss = sagan_discriminator_loss(real_output,fake_output)

  gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
  d_op.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


  #train and upate generator
  with tf.GradientTape() as gen_tape:
    generated_images = generator(noise, training=True)
    fake_output = discriminator(generated_images, training=True)
    gen_loss = sagan_generator_loss(fake_output)

  gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
  g_op.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

  return disc_loss, gen_loss

def train(args):

  if(args.no_scheduler):

    d_lr = args.d_lr
    g_lr = args.g_lr

  else:

    d_lr = tf.keras.optimizers.schedules.ExponentialDecay(args.d_lr, args.num_training_steps,0.96)
    g_lr = tf.keras.optimizers.schedules.ExponentialDecay(args.g_lr, args.num_training_steps,0.96)


  #setup optimizers
  discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=d_lr, beta_1=0.0,beta_2=0.9)
  generator_optimizer = tf.keras.optimizers.Adam(learning_rate=g_lr, beta_1=0.0, beta_2=0.9)
  

  #setup tf.dataset
  data_dir = args.data_dir
  
  dataset = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    seed = 123,
    image_size = (args.img_dim,args.img_dim),
    batch_size = args.batch_size,
    label_mode = None)

  normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset=-1)


  dataset = dataset.map(lambda x: normalization_layer(x))

  AUTOTUNE = tf.data.experimental.AUTOTUNE

  dataset = dataset.cache().prefetch(buffer_size=AUTOTUNE)

  #setup models
  noise_shape = (1,1,args.noise_dim)
  kernel_init = tf.keras.initializers.GlorotNormal()


  generator = make_sagan_generator_model(args.img_dim, noise_shape, args.gen_kernel_size, kernel_init)
  discriminator = make_sagan_discriminator_model(args.img_dim, args.disc_kernel_size, kernel_init)


  generator.summary()
  discriminator.summary()

  #setup reporting lists
  all_disc_loss = []
  all_gen_loss = []

  #seed values for reporting images
  latent_seed = tf.random.normal([9, 1,1,args.noise_dim])

  #start counter
  step_begin_time = time.time()


  #run through all steps using tf dataset

  #calculate epochs based off of desired number of steps
  files = sorted(glob.glob(data_dir + 'images/*.png')) 
  files_len = len(files)
  steps_per_epoch = files_len / args.batch_size
  num_epochs = int(args.num_training_steps / steps_per_epoch)

  step_counter = 0

  for epoch in range(num_epochs):

    for batch in dataset:

      noise = tf.random.normal([args.batch_size,1,1,args.noise_dim])

      #perform forward and backward passes
      disc_loss, gen_loss = train_step(discriminator,generator, discriminator_optimizer,generator_optimizer,batch, noise)
      all_disc_loss.append(disc_loss)
      all_gen_loss.append(gen_loss)

      #reporting
      if (step_counter % args.print_freq) == 0:

        end_time = time.time()
        diff_time = int(end_time - step_begin_time)

        print("Step %d completed. Time took: %s secs." % (step_counter, diff_time))

        generate_and_save_images(generator, step_counter, latent_seed)

        #plot_losses
        plot_loss(all_disc_loss,all_gen_loss)
        
        step_begin_time = time.time()
          
      #save models
      if(step_counter % args.save_freq == 0 and step_counter > 0):

        print("saving model at {}".format(step_counter))

        generator.save_weights("SavedModels/generator_weights_at_step_{}.h5".format(step_counter))
        discriminator.save_weights("SavedModels/discriminator_weights_at_step_{}.h5".format(step_counter))

      step_counter += 1

  generator.save_weights("SavedModels/generator_weights_at_step_{}.h5".format(args.num_training_steps))
  discriminator.save_weights("SavedModels/discriminator_weights_at_step_{}.h5".format(args.num_training_steps))

