# %%
# Updated model code
import tensorflow as tf
import librosa
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os

# Define TemporalBlock and TemporalConvNet classes
class TemporalBlock(tf.keras.Model):
    def __init__(self, dilation_rate, nb_filters, kernel_size, padding, dropout_rate=0.0):
        super(TemporalBlock, self).__init__()
        init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
        assert padding in ['causal', 'same']

        # block1
        self.conv1 = tf.keras.layers.Conv1D(filters=nb_filters, kernel_size=kernel_size,
                                            dilation_rate=dilation_rate, padding=padding, kernel_initializer=init)
        self.batch1 = tf.keras.layers.BatchNormalization(axis=-1)
        self.ac1 = tf.keras.layers.Activation('relu')
        self.drop1 = tf.keras.layers.Dropout(rate=dropout_rate)

        # block2
        self.conv2 = tf.keras.layers.Conv1D(filters=nb_filters, kernel_size=kernel_size,
                                            dilation_rate=dilation_rate, padding=padding, kernel_initializer=init)
        self.batch2 = tf.keras.layers.BatchNormalization(axis=-1)
        self.ac2 = tf.keras.layers.Activation('relu')
        self.drop2 = tf.keras.layers.Dropout(rate=dropout_rate)

        # 1*1 convolution to match dimensions
        self.downsample = tf.keras.layers.Conv1D(filters=nb_filters, kernel_size=1,
                                                 padding='same', kernel_initializer=init)
        self.ac3 = tf.keras.layers.Activation('relu')

    def call(self, x, training=False):
        prev_x = x
        x = self.conv1(x)
        x = self.batch1(x, training=training)
        x = self.ac1(x)
        x = self.drop1(x, training=training)

        x = self.conv2(x)
        x = self.batch2(x, training=training)
        x = self.ac2(x)
        x = self.drop2(x, training=training)

        if prev_x.shape[-1] != x.shape[-1]:    # match the dimension
            prev_x = self.downsample(prev_x)
        assert prev_x.shape == x.shape

        return self.ac3(prev_x + x)  # skip connection

class TemporalConvNet(tf.keras.Model):
    def __init__(self, num_channels, kernel_size=2, dropout=0.2):
        # num_channels is a list containing hidden sizes of Conv1D
        super(TemporalConvNet, self).__init__()
        assert isinstance(num_channels, list)

        model = tf.keras.Sequential()
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_rate = 2 ** i  # exponential growth
            model.add(TemporalBlock(dilation_rate, num_channels[i], kernel_size,
                                    padding='causal', dropout_rate=dropout))
        self.network = model

    def call(self, x, training=False):
        return self.network(x, training=training)

# Load the audio files
INPUT_FILE_DIR = "data/cab_tweedCabShort_off_gain_7.5.wav"
CONV_FILE_DIR = "data/tweedCabShort.flac"

FS = 44100
x = librosa.load(INPUT_FILE_DIR, sr=FS)[0]
kernel = librosa.load(CONV_FILE_DIR, sr=FS)[0]

# Normalize the input data
x = (x - np.mean(x)) / np.std(x)
kernel = (kernel - np.mean(kernel)) / np.std(kernel)

print(x.shape)
print(kernel.shape)

# Define the input layer for batches of 32 samples
input_A = tf.keras.layers.Input(shape=(32, 1), name='input_A')

# Apply TemporalConvNet
tcn_A = TemporalConvNet(num_channels=[16, 16, 16], kernel_size=3, dropout=0.2)(input_A)

# Add a pooling layer to reduce dimensionality
pooled_A = tf.keras.layers.GlobalAveragePooling1D()(tcn_A)

# Pass the pooled output through a dense layer M4 to produce the final output O
M4 = tf.keras.layers.Dense(32, activation='relu', name='M4')(pooled_A)
output_O = tf.keras.layers.Dense(32, activation='relu', name='output_O')(M4)

# Create the model
model = tf.keras.models.Model(inputs=input_A, outputs=output_O, name='M')

def custom_loss(y_true, y_pred):
    # Mean Squared Error (MSE)
    mse = tf.reduce_mean(tf.square(y_true - y_pred))

    # Error-to-Signal Ratio (ESR)
    esr = tf.reduce_mean(tf.square(y_true - y_pred)) / tf.reduce_mean(tf.square(y_true))

    # Cumulative Error (absolute sum of errors)
    cumulative_error = tf.reduce_sum(tf.abs(y_true - y_pred))

    return esr + mse + cumulative_error

# Custom learning rate schedule
class CustomLearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, decay_rate):
        self.initial_learning_rate = initial_learning_rate
        self.decay_rate = decay_rate

    def __call__(self, step):
        return self.initial_learning_rate / (1 + self.decay_rate * tf.cast(step, tf.float32))

initial_learning_rate = 0.001
decay_rate = 0.01
learning_rate_schedule = CustomLearningRateSchedule(initial_learning_rate, decay_rate)

# Custom optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_schedule)

# Compile the model with the custom loss function and custom optimizer
model.compile(optimizer=optimizer, loss=custom_loss)

# Summary of the model
model.summary()

# Plot
tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

# Import image and display
img = mpimg.imread('model.png')
imgplot = plt.imshow(img)
plt.show()

# Delete the image
os.remove('model.png')

# %%
# Convolve the two signals
y = np.convolve(x, kernel, mode='same')

# Create dataset by splitting into batches of 32 samples
def create_batches(data, batch_size):
    num_batches = len(data) // batch_size
    return np.array([data[i*batch_size:(i+1)*batch_size] for i in range(num_batches)])

# Split the data into training and testing sets (70% train, 30% test)
split_index = int(0.7 * len(x))

x_train = x[:split_index]
x_test = x[split_index:]
y_train = y[:split_index]
y_test = y[split_index:]

X_train_batches = create_batches(x_train, 32)
Y_train_batches = create_batches(y_train, 32)
X_test_batches = create_batches(x_test, 32)
Y_test_batches = create_batches(y_test, 32)

X_train_batches = np.expand_dims(X_train_batches, axis=-1)
X_test_batches = np.expand_dims(X_test_batches, axis=-1)

# Train the model on the training dataset
model.fit(X_train_batches, Y_train_batches, epochs=10)

# Evaluate the model on the testing dataset
loss = model.evaluate(X_test_batches, Y_test_batches)
print(f"Test Loss: {loss}")

# Predict the output for the test dataset
Y_pred_batches = model.predict(X_test_batches)

# Flatten the predictions and ground truth to compare
Y_pred = Y_pred_batches.flatten()
Y_true = Y_test_batches.flatten()

# Plot the predicted output against the ground truth
plt.figure(figsize=(14, 7))
plt.plot(Y_true, label='Ground Truth')
plt.plot(Y_pred, label='Predicted', alpha=0.7)
plt.legend()
plt.title('Predicted Output vs Ground Truth')
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.show()

# %%
