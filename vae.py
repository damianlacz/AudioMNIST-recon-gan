import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Lambda

class VAE(Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.mse_loss = MeanSquaredError()
        self.beta = 1
        self.reconstruction_loss_weight = 1

    @tf.function
    def sample(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    @tf.function
    def call(self, inputs):
        z_mean, z_log_var = self.encoder(inputs)
        z = Lambda(self.sample)([z_mean, z_log_var])
        reconstructed = self.decoder(z)
        return reconstructed

    @tf.function
    def compute_loss(self, inputs):
        z_mean, z_log_var = self.encoder(inputs)
        z = Lambda(self.sample)([z_mean, z_log_var])
        reconstructed = self.decoder(z)
        reconstruction_loss = self.mse_loss(inputs, reconstructed)
        kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
        total_loss = self.reconstruction_loss_weight * reconstruction_loss + self.beta * kl_loss
        return total_loss, self.beta * kl_loss, self.reconstruction_loss_weight * reconstruction_loss

    @tf.function
    def train_step(self, data):
        with tf.GradientTape() as tape:
            loss, kl_loss, reconstruction_loss = self.compute_loss(data)
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {"total_loss": loss,
                "kl_loss": kl_loss,
                "reconstruction_loss": reconstruction_loss}
    
    @tf.function
    def test_step(self, inputs):
        z_mean, z_log_var = self.encoder(inputs)
        z = Lambda(self.sample)([z_mean, z_log_var])
        reconstructed = self.decoder(z)
        reconstruction_loss = self.mse_loss(inputs, reconstructed)
        kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        total_loss = reconstruction_loss + kl_loss
        return {"total_loss" : total_loss}
