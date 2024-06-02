import tensorflow as tf
import numpy as np

class VAE(tf.keras.Model):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.models.Sequential()
        self.encoder.add(tf.keras.layers.InputLayer(shape = (11025, 1)))
        self.encoder.add(tf.keras.layers.Conv1D(filters = 8, kernel_size = 3, strides = 5, activation = 'softmax'))
        self.encoder.add(tf.keras.layers.Conv1D(filters = 32, kernel_size = 3, strides = 5, activation = 'softmax'))
        self.encoder.add(tf.keras.layers.Conv1D(filters = 64, kernel_size = 3, strides = 3, activation = 'softmax'))
        self.encoder.add(tf.keras.layers.Flatten())
        self.encoder.add(tf.keras.layers.Dense(latent_dim + latent_dim))

        self.decoder = tf.keras.models.Sequential()
        self.decoder.add(tf.keras.layers.InputLayer(shape = (latent_dim, )))
        self.decoder.add(tf.keras.layers.Dense(147*32, activation = 'relu'))
        self.decoder.add(tf.keras.layers.Reshape(target_shape = (147, 32)))
        self.decoder.add(tf.keras.layers.Conv1DTranspose(filters = 128, kernel_size = 3, strides = 3, activation = 'relu'))
        self.decoder.add(tf.keras.layers.Conv1DTranspose(filters = 32, kernel_size = 3, strides = 5, activation = 'relu'))
        self.decoder.add(tf.keras.layers.Conv1DTranspose(filters = 1, kernel_size = 3, strides = 5, activation = 'sigmoid'))

    @tf.function
    def sample(self, eps = None):
        if eps is None:
            eps = tf.random.normal(shape = (100, self.latent_dim))
        return self.decode(eps)

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits = 2, axis = 1)
        return mean, logvar
    
    def reparameterize(self, mean, logvar):
        return tf.random.normal(shape = mean.shape) * tf.exp(.5 * logvar) + mean

    def decode(self, z):
        return self.decoder(z)
    
    def train_step(self, x, optimizer):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x)
            gradients = tape.gradient(loss, self.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.trainable_variables))
    
    def compute_loss(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_recon = self.decode(z)
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits = x_recon, labels = x)
        logpx_z = -tf.reduce_sum(cross_ent, axis = [1, 2])
        logpz = self._log_normal_pdf(z, 0., 0.)
        logqz_x = self._log_normal_pdf(z, mean, logvar)
        return -tf.reduce_mean(logpx_z + logpz - logqz_x)

    
    def _log_normal_pdf(self, sample, mean, logvar, raxis = 1):
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum(-.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi), axis = raxis)