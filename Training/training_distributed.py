import tensorflow as tf
from Model.discriminator import make_sagan_discriminator_model
from Model.generator import make_sagan_generator_model
from Model.load_model import load_models_from_step
from Training.loss_functions import sagan_discriminator_loss, sagan_generator_loss
from Utils.reporting import generate_and_save_images, plot_loss
import time
import glob


class DistributedTrainer:

  def __init__(self, args):
    #define distribution strategy
    self.strategy = tf.distribute.MirroredStrategy()
    print ('Number of devices: {}'.format(self.strategy.num_replicas_in_sync))

    #training hyper parameters
    if(args.no_scheduler):
      self.d_lr = args.d_lr
      self.g_lr = args.g_lr
    else:
      self.d_lr = tf.keras.optimizers.schedules.ExponentialDecay(args.d_lr, args.num_training_steps,0.96)
      self.g_lr = tf.keras.optimizers.schedules.ExponentialDecay(args.g_lr, args.num_training_steps,0.96)

    self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=self.d_lr, beta_1=0.0,beta_2=0.9)
    self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=self.g_lr, beta_1=0.0, beta_2=0.9)

    self.global_batch_size = args.batch_size
    self.batch_size_per_replica = self.global_batch_size / self.strategy.num_replicas_in_sync

    self.num_training_steps = args.num_training_steps

    #architechture parameters
    self.noise_dim = args.noise_dim
    self.noise_shape = (1,1,self.noise_dim)
    self.kernel_init = tf.keras.initializers.GlorotNormal()
    self.img_dim = args.img_dim

    #dataset parameters
    self.data_dir = args.data_dir
    self.dataset = self._init_data_set()

    self.all_disc_loss = []
    self.all_gen_loss = []

  def _init_data_set(self):

    dataset = tf.keras.preprocessing.image_dataset_from_directory(
      self.data_dir,
      seed = 123,
      image_size = (self.img_dim,self.img_dim),
      batch_size = self.global_batch_size,
      label_mode = None)

    normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset=-1)

    dataset = dataset.map(lambda x: normalization_layer(x))

    AUTOTUNE = tf.data.experimental.AUTOTUNE

    dataset = dataset.cache().prefetch(buffer_size=AUTOTUNE)

    dataset = self.strategy.experimental_distribute_dataset(dataset)

    return dataset


  @tf.function
  def train_step(self, images):

    noise = tf.random.normal([images.shape[0],1,1,self.noise_dim])

    #train and upate discriminator on real data
    with tf.GradientTape() as disc_tape:
      real_output = self.discriminator(images, training=True)
      generated_images = self.generator(noise, training=True)
      fake_output = self.discriminator(generated_images, training=True)

      disc_loss = sagan_discriminator_loss(real_output,fake_output)

    gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
    self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))


    #train and upate generator
    with tf.GradientTape() as gen_tape:
      generated_images = self.generator(noise, training=True)
      fake_output = self.discriminator(generated_images, training=True)
      gen_loss = sagan_generator_loss(fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
    self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))

    return disc_loss, gen_loss


  @tf.function
  def distribute_trains_step(self, dist_dataset):
    per_replica_g_loss, per_replica_d_loss = self.strategy.run(self.train_step, args=(dist_dataset,))

    total_g_loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_g_loss, axis=None)
    total_d_loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_d_loss, axis=None)

    self.all_disc_loss.append(total_g_loss)
    self.all_gen_loss.append(total_d_loss)

    return total_g_loss, total_d_loss


  def train(self, args):

    #seed values for reporting images
    latent_seed = tf.random.normal([9, 1,1,self.noise_dim])

    #start counter
    step_begin_time = time.time()

    #calculate epochs based off of desired number of steps
    files = sorted(glob.glob(self.data_dir + 'images/*')) 
    files_len = len(files)
    steps_per_epoch = files_len / self.global_batch_size
    num_epochs = int(self.num_training_steps / steps_per_epoch)

    step_counter = 0

    with self.strategy.scope():

      self.generator = make_sagan_generator_model(self.img_dim, self.noise_shape, args.gen_kernel_size, self.kernel_init)
      self.discriminator = make_sagan_discriminator_model(self.img_dim, args.disc_kernel_size, self.kernel_init)
      self.generator.summary()
      # self.discriminator.summary()

      for epoch in range(num_epochs):

        for batch in self.dataset:

          #perform forward and backward passes
          disc_loss, gen_loss = self.distribute_trains_step(batch)

          #reporting
          if (step_counter % args.print_freq) == 0:

            end_time = time.time()
            diff_time = int(end_time - step_begin_time)

            print("Step %d completed. Time took: %s secs." % (step_counter, diff_time))

            generate_and_save_images(self.generator, step_counter, latent_seed)

            #plot_losses
            # plot_loss(self.all_disc_loss,self.all_gen_loss)
            
            step_begin_time = time.time()
              
          #save models
          if(step_counter % args.save_freq == 0 and step_counter > 0):

            print("saving model at {}".format(step_counter))

            self.generator.save_weights("SavedModels/generator_weights_at_step_{}.h5".format(step_counter))
            self.discriminator.save_weights("SavedModels/discriminator_weights_at_step_{}.h5".format(step_counter))

          step_counter += 1

    self.generator.save_weights("SavedModels/generator_weights_at_step_{}.h5".format(args.num_training_steps))
    self.discriminator.save_weights("SavedModels/discriminator_weights_at_step_{}.h5".format(args.num_training_steps))

