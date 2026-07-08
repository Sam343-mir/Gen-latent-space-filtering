import tensorflow as tf
from keras import layers
from keras import backend as K
from tensorflow import keras
import numpy as np

class ConvAutoencoder:
    @staticmethod
    def build(width, height, depth, latentDim=256):

        inputs = keras.Input(shape=(height, width, depth))

        x = layers.Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(inputs)
        x = layers.LeakyReLU()(x)
        x = layers.Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(x)
        x = layers.LeakyReLU()(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        
        x = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(x)
        x = layers.LeakyReLU()(x)
        x = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(x)
        x = layers.LeakyReLU()(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        
        x = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(x)
        x = layers.LeakyReLU()(x)
        x = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(x)
        x = layers.LeakyReLU()(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        
        # x = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(x)
        # x = layers.LeakyReLU()(x)
        # x = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(x)
        # x = layers.LeakyReLU()(x)
        # x = layers.MaxPooling2D((2, 2), padding='same')(x)

        # volumeSize = K.int_shape(x)
        volumeSize = x.shape

        flatten = layers.Flatten()(x)
        latent = layers.Dense(latentDim)(flatten)
        encoder = tf.keras.Model(inputs, latent, name="Encoder")

        latentInputs = keras.Input(shape=(latentDim,))
        x = layers.Dense(np.prod(volumeSize[1:]))(latentInputs)
        x = layers.Reshape((volumeSize[1], volumeSize[2], volumeSize[3]))(x)

        # latentInputs = keras.Input(shape=(latentDim,))

        # x = layers.Dense(8 * 8 * 8)(latentInputs)
        # x = layers.Reshape((8,8,8), name='Reshape')(x)

        x = layers.Conv2DTranspose(64, (3, 3), strides=2, padding="same", kernel_initializer='he_normal')(x)
        x = layers.LeakyReLU()(x)
        x = layers.Conv2D(64, (3, 3), padding="same", kernel_initializer='he_normal')(x)
        x = layers.LeakyReLU()(x)
        
        # x = layers.Conv2DTranspose(64, (3, 3), strides=2, padding="same", kernel_initializer='he_normal')(x)
        # x = keras.activations.tanh(x)
        # x = layers.Conv2D(64, (3, 3), padding="same", kernel_initializer='he_normal')(x)
        # x = keras.activations.tanh(x)
        
        x = layers.Conv2DTranspose(64, (3, 3), strides=2, padding="same", kernel_initializer='he_normal')(x)
        x = layers.LeakyReLU()(x)
        x = layers.Conv2D(64, (3, 3), padding="same", kernel_initializer='he_normal')(x)
        x = layers.LeakyReLU()(x)

        x = layers.Conv2DTranspose(32, (3, 3), strides=2, padding="same", kernel_initializer='he_normal')(x)
        x = layers.LeakyReLU()(x)
        x = layers.Conv2D(32, (3, 3), padding="same", kernel_initializer='he_normal')(x)
        x = layers.LeakyReLU()(x)

        outputs = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

        decoder = keras.Model(latentInputs, outputs, name="decoder")

        # our autoencoder is the encoder + decoder
        autoencoder = keras.Model(inputs, decoder(encoder(inputs)),
			name="autoencoder")

		# return a 3-tuple of the encoder, decoder, and autoencoder
        return (encoder, decoder, autoencoder)

        # inputs = keras.Input(shape=(height, width, depth))

        # x = layers.Conv2D(16, (3, 3), padding='same')(inputs)
        # x = layers.MaxPooling2D((2, 2), padding='same')(x)
        # # x = layers.BatchNormalization()(x)
        # x = layers.LeakyReLU()(x)
        # x = layers.Conv2D(8, (3, 3), padding='same')(x)
        # x = layers.MaxPooling2D((2, 2), padding='same')(x)
        # # x = layers.BatchNormalization()(x)
        # x = layers.LeakyReLU()(x)
        # x = layers.Conv2D(8, (3, 3), padding='same')(x)
        # x = layers.MaxPooling2D((2, 2), padding='same')(x)
        # # x = layers.BatchNormalization()(x)
        # x = layers.LeakyReLU()(x)
        # x = layers.Conv2D(8, (3, 3), padding='same')(x)
        # # x = layers.MaxPooling2D((2, 2), padding='same')(x)
        # # x = layers.Conv2D(8, (3, 3), activation='tanh', padding='same')(x)
        # x = layers.MaxPooling2D((2, 2), padding='same')(x)
        # # x = layers.BatchNormalization()(x)
        # x = layers.LeakyReLU()(x)
        # x = layers.Conv2D(8, (3, 3), padding='same')(x)
        # x = layers.LeakyReLU()(x)
        # # encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

        # flatten = layers.Flatten()(x)
        # latent = layers.Dense(latentDim)(flatten)
        # encoder = tf.keras.Model(inputs, latent, name="Encoder")


        # latentInputs = keras.Input(shape=(latentDim,))

        # x = layers.Dense(8 * 8 * 8)(latentInputs)
        # x = layers.Reshape((8,8,8), name='Reshape')(x)

        # x = layers.Conv2D(8, (3, 3), padding='same')(x)
        # x = layers.UpSampling2D((2, 2))(x)
        # # x = layers.Conv2D(8, (3, 3), activation='tanh', padding='same')(x)
        # # x = layers.UpSampling2D((2, 2))(x)
        # # x = layers.BatchNormalization()(x)
        # x = layers.LeakyReLU()(x)
        # x = layers.Conv2D(8, (3, 3), padding='same')(x)
        # x = layers.UpSampling2D((2, 2))(x)
        # # x = layers.BatchNormalization()(x)
        # x = layers.Conv2D(8, (3, 3), padding='same')(x)
        # x = layers.UpSampling2D((2, 2))(x)
        # # x = layers.BatchNormalization()(x)
        # x = layers.LeakyReLU()(x)
        # x = layers.Conv2D(8, (3, 3), padding='same')(x)
        # x = layers.UpSampling2D((2, 2))(x)
        # # x = layers.BatchNormalization()(x)
        # x = layers.LeakyReLU()(x)
        # x = layers.Conv2D(16, (3, 3), padding='same')(x)
        # x = layers.UpSampling2D((2, 2))(x)
        # x = layers.LeakyReLU()(x)
        # outputs = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
        
        # decoder = tf.keras.Model(latentInputs, outputs, name="decoder")

        # # our autoencoder is the encoder + decoder
        # autoencoder = tf.keras.Model(inputs, decoder(encoder(inputs)),
		# 	name="autoencoder")

		# # return a 3-tuple of the encoder, decoder, and autoencoder
        # return (encoder, decoder, autoencoder)
