import tensorflow as tf

def sagan_discriminator_loss(real, fake):

    real_loss = tf.reduce_mean(tf.nn.relu(1.0 - real))
    fake_loss = tf.reduce_mean(tf.nn.relu(1.0 + fake))

    loss = real_loss + fake_loss

    return loss

def sagan_generator_loss(fake):

    fake_loss = -tf.reduce_mean(fake)

    loss = fake_loss

    return loss