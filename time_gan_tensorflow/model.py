import numpy as np
import tensorflow as tf
import os

# Check for GPU availability
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Using {len(gpus)} GPU(s): {[gpu.name for gpu in gpus]}")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU found. Running on CPU.")

from time_gan_tensorflow.utils import time_series_to_sequences, sequences_to_time_series
from time_gan_tensorflow.modules import (
    encoder_embedder, encoder, decoder, generator_embedder, generator, discriminator, simulator
)
from time_gan_tensorflow.losses import binary_crossentropy, mean_squared_error

class TimeGAN():
    def __init__(self, x, timesteps, hidden_dim, num_layers, lambda_param, eta_param, learning_rate, batch_size):
        """
        Implementation of TimeGAN with GPU acceleration.
        """
        self.device = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"

        with tf.device(self.device):
            samples = x.shape[0]
            features = x.shape[1]

            # Normalize data
            self.mu = np.mean(x, axis=0)
            self.sigma = np.std(x, axis=0)
            x = (x - self.mu) / self.sigma

            # Convert time series into sequences
            x = time_series_to_sequences(time_series=x, timesteps=timesteps)

            # Create the TensorFlow dataset
            self.dataset = (tf.data.Dataset.from_tensor_slices(x)
                            .cache()
                            .shuffle(samples)
                            .batch(batch_size)
                            .prefetch(tf.data.AUTOTUNE))

            # Build models
            self.autoencoder_model = tf.keras.Sequential([
                encoder_embedder(timesteps=timesteps, features=features, hidden_dim=hidden_dim, num_layers=1),
                encoder(timesteps=timesteps, hidden_dim=hidden_dim, num_layers=num_layers - 1),
                decoder(timesteps=timesteps, features=features, hidden_dim=hidden_dim, num_layers=num_layers)
            ])

            self.generator_model = tf.keras.Sequential([
                generator_embedder(timesteps=timesteps, features=features, hidden_dim=hidden_dim, num_layers=1),
                generator(timesteps=timesteps, hidden_dim=hidden_dim, num_layers=num_layers - 1),
            ])

            self.discriminator_model = discriminator(timesteps=timesteps, hidden_dim=hidden_dim, num_layers=num_layers)

            # Optimizers
            self.autoencoder_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

            # Store parameters
            self.samples = samples
            self.timesteps = timesteps
            self.features = features
            self.lambda_param = lambda_param
            self.eta_param = eta_param

    def fit(self, epochs, verbose=True):
        """
        Train TimeGAN using TensorFlow and GPU acceleration.
        """

        @tf.function
        def train_step(data):
            with tf.device(self.device):  # Ensure computations run on GPU
                with tf.GradientTape() as autoencoder_tape, tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:
                    x = tf.cast(data, dtype=tf.float32)
                    z = simulator(samples=tf.shape(x)[0], timesteps=self.timesteps, features=self.features)

                    # Encoder outputs
                    ex = self.autoencoder_model.get_layer('encoder_embedder')(x)
                    hx = self.autoencoder_model.get_layer('encoder')(ex)

                    # Generator outputs
                    ez = self.generator_model.get_layer('generator_embedder')(z)
                    hz = self.generator_model.get_layer('generator')(ez)
                    hx_hat = self.generator_model.get_layer('generator')(ex)

                    # Decoder outputs
                    x_hat = self.autoencoder_model.get_layer('decoder')(hx)

                    # Discriminator outputs
                    p_ex = self.discriminator_model(ex)
                    p_ez = self.discriminator_model(ez)
                    p_hx = self.discriminator_model(hx)
                    p_hz = self.discriminator_model(hz)

                    # Loss calculations
                    supervised_loss = mean_squared_error(hx[:, 1:, :], hx_hat[:, :-1, :])
                    autoencoder_loss = mean_squared_error(x, x_hat) + self.lambda_param * supervised_loss
                    generator_loss = (
                        binary_crossentropy(tf.ones_like(p_hz), p_hz) +
                        binary_crossentropy(tf.ones_like(p_ez), p_ez) +
                        self.eta_param * supervised_loss
                    )
                    discriminator_loss = (
                        binary_crossentropy(tf.zeros_like(p_hz), p_hz) +
                        binary_crossentropy(tf.zeros_like(p_ez), p_ez) +
                        binary_crossentropy(tf.ones_like(p_hx), p_hx) +
                        binary_crossentropy(tf.ones_like(p_ex), p_ex)
                    )

                # Compute gradients
                autoencoder_grad = autoencoder_tape.gradient(autoencoder_loss, self.autoencoder_model.trainable_variables)
                generator_grad = generator_tape.gradient(generator_loss, self.generator_model.trainable_variables)
                discriminator_grad = discriminator_tape.gradient(discriminator_loss, self.discriminator_model.trainable_variables)

                # Apply gradients
                self.autoencoder_optimizer.apply_gradients(zip(autoencoder_grad, self.autoencoder_model.trainable_variables))
                self.generator_optimizer.apply_gradients(zip(generator_grad, self.generator_model.trainable_variables))
                self.discriminator_optimizer.apply_gradients(zip(discriminator_grad, self.discriminator_model.trainable_variables))

                return autoencoder_loss, generator_loss, discriminator_loss

        for epoch in range(epochs):
            for data in self.dataset:
                autoencoder_loss, generator_loss, discriminator_loss = train_step(data)
            if verbose:
                print(
                    f'Epoch {epoch + 1} '
                    f'Autoencoder Loss: {autoencoder_loss.numpy():.6f} '
                    f'Generator Loss: {generator_loss.numpy():.6f} '
                    f'Discriminator Loss: {discriminator_loss.numpy():.6f}'
                )

    def reconstruct(self, x):
        """
        Reconstruct the input time series using the trained autoencoder.
        """
        x = (x - self.mu) / self.sigma
        x = time_series_to_sequences(time_series=x, timesteps=self.timesteps)
        x_hat = self.autoencoder_model(x)
        x_hat = sequences_to_time_series(x_hat.numpy())
        return self.mu + self.sigma * x_hat

    def simulate(self, samples):
        """
        Generate synthetic time series data using the trained generator.
        """
        z = simulator(samples=samples // self.timesteps, timesteps=self.timesteps, features=self.features)
        x_sim = self.autoencoder_model.get_layer('decoder')(self.generator_model(z))
        x_sim = sequences_to_time_series(x_sim.numpy())
        return self.mu + self.sigma * x_sim

    def save_model(self, model_name, path='models'):
        """
        Save the trained models using the `.keras` format.
        """
        os.makedirs(f'{path}/{model_name}', exist_ok=True)

        # Save in the recommended `.keras` format
        self.autoencoder_model.save(f'{path}/{model_name}/autoencoder_model.keras')
        self.generator_model.save(f'{path}/{model_name}/generator_model.keras')
        self.discriminator_model.save(f'{path}/{model_name}/discriminator_model.keras')

        # Save normalization parameters
        np.save(f'{path}/{model_name}/mu.npy', self.mu)
        np.save(f'{path}/{model_name}/sigma.npy', self.sigma)


    def load_model(self, model_name, path='models'):
        """
        Load trained models from disk.
        """
        self.autoencoder_model = tf.keras.models.load_model(f'{path}/{model_name}/autoencoder_model.keras', compile=False)
        self.generator_model = tf.keras.models.load_model(f'{path}/{model_name}/generator_model.keras', compile=False)
        self.discriminator_model = tf.keras.models.load_model(f'{path}/{model_name}/discriminator_model.keras', compile=False)

        # Load normalization parameters
        self.mu = np.load(f'{path}/{model_name}/mu.npy')
        self.sigma = np.load(f'{path}/{model_name}/sigma.npy')

        # Recompile the models after loading
        self.autoencoder_model.compile(optimizer=self.autoencoder_optimizer)
        self.generator_model.compile(optimizer=self.generator_optimizer)
        self.discriminator_model.compile(optimizer=self.discriminator_optimizer)

