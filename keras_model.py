import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer, Dropout, Conv1D, Conv2D, Flatten, Reshape, MaxPooling1D, MaxPooling2D, BatchNormalization, TimeDistributed, ReLU
from tensorflow.keras.optimizers import Adam, SGD

# model architecture
model = Sequential()
channels = 1
columns = 13
rows = int(input_length / (columns * channels))
model.add(Reshape((rows, columns, channels), input_shape=(input_length, )))
model.add(Conv2D(12, kernel_size=3, strides=2, activation='relu', padding='valid'))
# model.add(Dropout(0.5))
model.add(Conv2D(30, kernel_size=3, strides=2, activation=None, padding='valid'))
model.add(BatchNormalization()) # need to change?
model.add(ReLU())
# model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))
# model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(classes, activation='softmax',name='y_logis'))

# this controls the learning rate
opt = SGD(learning_rate=0.005, momentum= 0.6)
# this controls the batch size, or you can manipulate the tf.data.Dataset objects yourself
BATCH_SIZE = 1
train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=False)
validation_dataset = validation_dataset.batch(BATCH_SIZE, drop_remainder=False)
callbacks.append(BatchLoggerCallback(BATCH_SIZE, train_sample_count))

# train the neural network
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.fit(train_dataset, epochs=10, validation_data=validation_dataset, verbose=2, callbacks=callbacks)

# Use this flag to disable per-channel quantization for a model.
# This can reduce RAM usage for convolutional models, but may have
# an impact on accuracy.
disable_per_channel_quantization = False
