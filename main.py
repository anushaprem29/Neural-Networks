from __future__ import absolute_import, division, print_function
import tensorflow as tf
from tensorflow import keras
import datamanipulations

(train_data, train_labels), (test_data, test_labels) = datamanipulations.load_data()

print("Training set: {}".format(train_data.shape))  # 2010 examples, 10 features
print("Testing set:  {}".format(train_labels.shape))  # 501 examples, 10 features


def build_model():
    model = keras.Sequential([
        keras.layers.Dense(64, activation=tf.nn.relu,
                           input_shape=(train_data.shape[1],)),
        keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dense(4)
    ])

    optimizer = tf.train.RMSPropOptimizer(0.001)

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae'])
    return model


class print_epoch_info(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        print('Epoch ', epoch, '/', EPOCHS)
        print('loss: ', logs['loss'], ' - mean_absolute_error: ',
              logs['mean_absolute_error'], ' - val_loss: ', logs['val_loss'],
              ' - val_mean_absolute_error', logs['val_mean_absolute_error'])


model1 = build_model()
model1.summary()
EPOCHS = 400
history1 = model1.fit(train_data, train_labels, epochs=EPOCHS,
                      validation_split=0.2, verbose=0,
                      callbacks=[print_epoch_info()])
[loss, mae] = model1.evaluate(test_data, test_labels, verbose=0)

print("Testing set Mean Abs Error: {:7.2f}".format(mae * 1000))
