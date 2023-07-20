# Loading  dataset
import tensorflow as tf
import numpy as np

from tensorflow.python.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.python.keras.models import Model
import matplotlib.pyplot as plt

# Loading  dataset
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

# Add noise to the images
noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

# Clip the images to the valid pixel range
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

# Normalize the pixel values
x_train_noisy = x_train_noisy / 255.
x_test_noisy = x_test_noisy / 255.

# Define the input layer
input_layer = Input(shape=(28, 28, 1))

# Encoder
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# Decoder
x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

# Define the model
autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Train the model
autoencoder.fit(x_train_noisy, x_train, epochs=10, batch_size=128,
validation_data=(x_test_noisy, x_test))

# Using model to denoise the images
denoised_images = autoencoder.predict(x_test_noisy)

# Display the first image in the test set
plt.imshow(x_test[0], cmap='gray')
plt.show()

# Display the first image in the noisy test set
plt.imshow(x_test_noisy[0], cmap='gray')
plt.show()

# Display the first image in the denoised test set
plt.imshow(denoised_images[4].squeeze(), cmap='gray')
plt.show()