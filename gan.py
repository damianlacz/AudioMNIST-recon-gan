import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy

class GAN(tf.keras.Model):
    def __init__(self, generator, discriminator, **kwargs):
        super(GAN, self).__init__()

        self.generator = generator
        self.discriminator = discriminator
        self.binary_cross_entropy = BinaryCrossentropy(from_logits = True)

        self.gen_loss_weight = 1
        self.real_loss_noise_weight = 0.15
        self.real_loss_weight = 1
        self.fake_loss_noise_weight = 0.15
        self.fake_loss_weight = 1

        self.batch_size = kwargs.get('batch_size', 32)
        self.gen_opt = tf.keras.optimizers.Adam(1e-3)
        self.disc_opt = tf.keras.optimizers.Adam(1e-4)

    def compute_loss(self, inputs):
        noise = tf.random.normal(shape = (self.batch_size, 1024, ))
        fake_result = self.discriminator(self.generator(noise, training = True), training = True)
        real_result = self.discriminator(inputs, training = True)

        real = tf.ones_like(real_result) - self.real_loss_noise_weight * tf.random.uniform(shape = tf.shape(real_result), 
                                                                                           minval = 0, maxval = 1)
        fake = tf.zeros_like(fake_result) + self.fake_loss_noise_weight * tf.random.uniform(shape = tf.shape(fake_result), 
                                                                                            minval = 0, maxval = 1)

        real_loss = self.real_loss_weight * self.binary_cross_entropy(real, real_result)
        fake_loss = self.fake_loss_weight * self.binary_cross_entropy(fake, fake_result)

        gen_loss = self.gen_loss_weight * self.binary_cross_entropy(tf.ones_like(fake_result), fake_result)
        disc_loss = real_loss + fake_loss

        return gen_loss, disc_loss, real_loss, fake_loss
    @tf.function
    def train_step(self, data):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_loss, disc_loss, real_loss, fake_loss = self.compute_loss(data)
        gen_grads = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        disc_grads = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.gen_opt.apply_gradients(zip(gen_grads, self.generator.trainable_variables))
        self.disc_opt.apply_gradients(zip(disc_grads, self.discriminator.trainable_variables))
        return {
            'gen_loss': gen_loss,
            'disc_loss': disc_loss,
            'real_loss': real_loss,
            'fake_loss': fake_loss
        }
        

    def predict(self, x):
        return self.generator(x, training = False)