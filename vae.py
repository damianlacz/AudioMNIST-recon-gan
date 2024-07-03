import tensorflow as tf
from tensorflow.keras.losses import KLDivergence, MeanSquaredError
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Lambda

class VAE(Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.mse_loss = MeanSquaredError()
        self.kl_loss = KLDivergence()
        self._beta = 1
        self._reconstruction_loss_weight = 10**4

    def call(self, inputs):
        z_mean, z_log_var = self.encoder(inputs)
        z = self.sampling([z_mean, z_log_var])
        reconstructed = self.decoder(z)
        return reconstructed

    def compute_loss(self, inputs):
        z_mean, z_log_var = self.encoder(inputs)
        z = Lambda(self.sample)([z_mean, z_log_var])
        reconstructed = self.decoder(z)
        reconstruction_loss = self._reconstruction_loss_weight * self.mse_loss(inputs, reconstructed)
        kl_loss = self._beta * self.kl_loss(inputs, reconstructed)
        total_loss = reconstruction_loss + kl_loss
        return total_loss, kl_loss, reconstruction_loss
    
    def sample(self, inputs):
        mean, logvar = inputs
        batch = tf.shape(mean)[0]
        dim = tf.shape(mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return mean + tf.exp(0.5 * logvar) * epsilon

    @tf.function
    def train_step(self, data):
        with tf.GradientTape() as tape:
            loss, kl_loss, reconstruction_loss = self.compute_loss(data)
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {"total_loss": loss,
                "kl_loss": kl_loss,
                "reconstruction_loss": reconstruction_loss}
    
    def test_step(self, inputs):
        z_mean, z_log_var = self.encoder(inputs)
        z = Lambda(self.sample)([z_mean, z_log_var])
        reconstructed = self.decoder(z)
        reconstruction_loss = self._beta * self.mse_loss(inputs, reconstructed)
        kl_loss = self._reconstruction_loss_weight * self.kl_loss
        total_loss = reconstruction_loss + kl_loss
        return {"total_loss" : total_loss}
