import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt


def simple_rnn(rnn_units):
    return tf.keras.layers.SimpleRNN(
        rnn_units,
        activation='relu',
        kernel_initializer='glorot_uniform',
        dropout=.4,
        return_sequences=False,
        stateful=True
    )


def build_model(rnn_units, batch_size, training=True):
    sequential = tf.keras.Sequential([
        tf.keras.layers.LSTM(rnn_units,dropout=.4),
        tf.keras.layers.Dense(20),
        tf.keras.layers.Dense(4)
    ])
    inputs = tf.keras.Input((seq_length, 6), batch_size)
    if training:
        x = preprocessing_layer(inputs)
    else:
        x = tf.keras.layers.experimental.preprocessing.Normalization()(inputs)
    outputs = sequential(x)
    model = tf.keras.Model(inputs, outputs)

    return model


def train_model(train, val):
    model = build_model(rnn_units, batch_size=64)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, verbose=1,
        mode='min', restore_best_weights=True
    )

    model.compile(loss='mean_absolute_error',
                  optimizer=tf.optimizers.Adam(learning_rate=4e-3),
                  metrics=['mean_squared_error', 'accuracy'])
    history = model.fit(x=train, batch_size=None,
                        epochs=500,
                        callbacks=[early_stopping],
                        validation_data=val,
                        validation_freq=1,
                        verbose=1)
    plot_history(history)
    filepath = os.path.join('./training_checkpoints', 'mymodel')
    model.save_weights(filepath, overwrite=True)


def plot_history(history):
    plt.figure(1)
    plt.plot(history.history['loss'], label='MAE (training data)')
    plt.plot(history.history['val_loss'], label='MAE (validation data)')
    plt.title('MAE for Cosine and Sine of Attitude')
    plt.ylabel('MAE value')
    plt.xlabel('Epoch')
    plt.legend(loc="lower left")
    plt.show()

    plt.figure(2)
    plt.plot(history.history['accuracy'], label='Accuracy (training data)')
    plt.plot(history.history['val_accuracy'], label='Accuracy (validation data)')
    plt.title('Accuracy for Cosine and Sine of Attitude')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc="upper left")
    plt.show()


def test_model(data):
    model = build_model(rnn_units, batch_size=1,training=False)
    model.load_weights('./training_checkpoints/mymodel')
    model.summary()

    outputs = []
    for seq in data:
        output = ((model(seq)).numpy())[0]
        output = np.asarray([
            np.arctan2(output[0], output[1]),
            np.arctan2(output[2], output[3])
        ])
        output *= 180 / np.pi
        outputs.append(output)
    return outputs


def adapt(x):
    # set weights for normalization of inputs
    global preprocessing_layer
    preprocessing_layer = tf.keras.layers.experimental.preprocessing.Normalization()
    preprocessing_layer.adapt(x)


def normalize_output(y):
    # normalize angles using cosine
    # ensures all values in [-1,1] and no large loss between -180 and 180 degrees
    y = y * np.pi / 180
    normalized_y = [[np.sin(row[0]),
                     np.cos(row[0]),
                     np.sin(row[1]),
                     np.cos(row[1])]
                    for row in y]

    return np.asarray(normalized_y)


def get_data(training=True):
    data = [[], []]
    if training:
        dir = '../data/training/'
    else:
        dir = '../data/testing/'
    for file_name in os.listdir(dir):
        inFile = open(dir + file_name, 'r')
        for line in inFile:
            if line.startswith('//') or line.startswith('PacketCounter'):
                continue
            values = line.split('	')
            values = [float(x) for x in values]
            data[0].append([values[i] for i in range(2, 8)])
            data[1].append([values[15], values[16]])

    x = np.asarray(data[0])
    y = normalize_output(np.asarray(data[1]))

    if training:
        len_val = len(data[0]) // 5

        # separate data into training and validation sets
        x_train = x[:-len_val]
        y_train = y[:-len_val]
        x_val = x[-len_val:]
        y_val = y[-len_val:]

        # adapt input normalization layer using training data
        adapt(x_train)

        train_ds = make_dataset(x_train, y_train)
        val_ds = make_dataset(x_val, y_val)

        return train_ds, val_ds
    else:
        ds = make_dataset(x)
        return ds


def make_dataset(x, y=None):
    if y is not None:
        y = y[seq_length-1:]

        batch_size = 64
        remainder = (len(x) - seq_length + 1) % batch_size
        x = x[:-remainder]

        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=x,
            targets=y,
            sequence_length=seq_length,
            batch_size=batch_size)
    else:
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=x,
            targets=None,
            sequence_length=seq_length,
            batch_size=1)
    return ds


def set_global():
    global seq_length
    seq_length = 200

    global rnn_units
    rnn_units = 20


if __name__ == '__main__':
    set_global()

    train, val = get_data(training=True)
    train_model(train, val)

    # inputs = get_data(training=False)
    # outputs = test_model(inputs)
    pass
