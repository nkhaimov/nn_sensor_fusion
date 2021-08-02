import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import keras_tuner as kt

# global
seq_length = 100
rnn_units = 20
preprocessing_layer = tf.keras.layers.experimental.preprocessing.Normalization()


def simple_rnn(rnn_units):
    """
    :param rnn_units: (integer) number of rnn_units in the simple_rnn
    :return: SimpleRNN layer with rnn_units
    """
    return tf.keras.layers.SimpleRNN(
        rnn_units,
        activation='relu',
        kernel_initializer='glorot_uniform',
        dropout=.4,
        return_sequences=False,
    )


def build_model(rnn_units):
    """
    :param rnn_units: (integer) number of neurons in the rnn layer
    :return: neural network model
    """
    sequential = tf.keras.Sequential([
        tf.keras.layers.LSTM(
            rnn_units,activation='relu', recurrent_activation='sigmoid',
            dropout=.14, recurrent_dropout=.2
        ),
        tf.keras.layers.Dense(4)
    ])
    inputs = tf.keras.Input((seq_length, 6))
    x = preprocessing_layer(inputs)
    outputs = sequential(x)
    model = tf.keras.Model(inputs, outputs)

    return model


def build_model_hp(hp):
    sequential = tf.keras.Sequential([
        tf.keras.layers.LSTM(
            units=hp.Int('rnn_units', 10, 100, step=5),
            activation=hp.Choice('activation', ['tanh', 'relu', 'sigmoid', 'linear']),
            recurrent_activation=hp.Choice('recurrent_activation', ['tanh','sigmoid', 'relu', 'linear']),
            dropout=hp.Float('dropout', 0.1, 0.6, step=.01),
            recurrent_dropout=hp.Float('recurrent_dropout', 0.1, 0.6, step=.01)
        )
    ])
    hp_dense = hp.Int('num_dense_layers', 0, 2, default=1)
    if hp_dense > 0:
        sequential.add(tf.keras.layers.Dense(
            hp.Int('dense_units1', 4, 100, step=5, parent_name='num_dense_layers', parent_values=[1,2]),
            hp.Choice('dense_activation1', ['linear', 'sigmoid', 'relu', 'tanh'],
                      parent_name='num_dense_layers', parent_values=[1,2])
        ))
        if hp_dense > 1:
            sequential.add(tf.keras.layers.Dense(
                hp.Int('dense_units2', 4, 100, step=5, parent_name='num_dense_layers', parent_values=[2]),
                hp.Choice('dense_activation2', ['linear', 'sigmoid', 'relu', 'tanh'],
                          parent_name='num_dense_layers', parent_values=[2])
            ))
    sequential.add(tf.keras.layers.Dense(
        4, hp.Choice('output_activation', ['linear', 'sigmoid', 'relu', 'tanh'])
    ))

    inputs = tf.keras.Input((seq_length, 6))
    x = preprocessing_layer(inputs)
    outputs = sequential(x)
    model = tf.keras.Model(inputs, outputs)

    hp_learning_rate = hp.Float('learning_rate', 1e-5, 5e-3, step=1e-5)
    opt = hp.Choice('optimizer', ['adam', 'sgd', 'rmsprop', 'nadam'])
    opt = tf.keras.optimizers.get(opt)
    opt.learning_rate = hp_learning_rate

    model.compile(loss='mean_absolute_error',
                  optimizer=opt,
                  metrics=['mean_squared_error', 'accuracy'])
    return model


def tune_hyperparameters(train, val):
    tuner = kt.Hyperband(
        build_model_hp,
        objective=kt.Objective('val_accuracy', direction='max'),
        max_epochs=40,
        hyperband_iterations=1)  # recommended to set as high as possible

    tuner.search(train,
                 validation_data=val,
                 epochs=40,
                 callbacks=[tf.keras.callbacks.EarlyStopping(patience=1), tf.keras.callbacks.TerminateOnNaN()])
    # best_model = tuner.get_best_models(1)[0]
    # filepath = os.path.join('./training_checkpoints', file_name)
    # best_model.save(filepath)

    tuner.results_summary(num_trials=10)


def train_model(train, val, file_name):
    """
    Function to train the model, save weights, and graph history (loss and metrics).
    :param train: (tensorflow Dataset) training dataset
    :param val: (tensorflow Dataset) validation dataset
    :param file_name: (string)
    :return: None
    """
    model = build_model(rnn_units)

    # Stop training after 5 epochs with no improvement in loss for validation set. Restore model weights to those
    # corresponding to lowest val_loss.
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=2, verbose=1,
        mode='min', restore_best_weights=True
    )

    # Set loss, optimizer, and metrics for model.
    model.compile(loss='mean_absolute_error',
                  optimizer=tf.optimizers.Nadam(learning_rate=.00222),
                  metrics=['mean_squared_error', 'accuracy'])

    # Train model. Validate after each epoch.
    history = model.fit(x=train, batch_size=None,
                        epochs=500,
                        callbacks=[early_stopping],
                        validation_data=val,
                        validation_freq=1,
                        verbose=1)

    # Plot loss and accuracy.
    plot_history(history)

    # Save model weights
    filepath = os.path.join('./training_checkpoints', file_name)
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
    plt.legend(loc="lower left")
    plt.show()


def test_model(data, file_name):
    """
    Load model weights. Run trained model on a set of inputs and return outputs scaled to angles.
    :param data: (tensorflow Dataset) inputs to run through the model
    :param file_name: (string) location of model weights to load from
    :return: (list) roll and pitch in degrees
    """
    model = build_model(rnn_units)
    model.load_weights('./training_checkpoints/' + file_name)
    model.summary()

    outputs = []
    for batch in data:
        output = (model(batch)).numpy()
        for item in output:
            # calculate angles using their cos and sin values
            angles = np.asarray([
                np.arctan2(item[0], item[1]),
                np.arctan2(item[2], item[3])
            ])
            angles *= 180 / np.pi
            outputs.append(angles)
    return np.asarray(outputs)


def graph_model_output(output, ref):
    ref = ref[seq_length - 1:]

    roll, pitch = output[:, 0], output[:, 1]
    roll_ref, pitch_ref = ref[:, 0], ref[:, 1]
    plt.figure(1)
    plt.plot(roll, label='model output', lw=0.5)
    plt.plot(roll_ref, label='reference', lw=0.5)
    plt.title('Roll')
    plt.legend(loc="lower left")
    plt.show()

    plt.figure(2)
    plt.plot(pitch, label='model output', lw=0.5)
    plt.plot(pitch_ref, label='reference', lw=0.5)
    plt.title('Pitch')
    plt.legend(loc="lower left")
    plt.show()

    roll_error = roll - roll_ref
    pitch_error = pitch - pitch_ref
    plt.figure(3)
    plt.plot(roll_error, label='roll error')
    plt.plot(pitch_error, label='pitch error')
    plt.title('Model error')
    plt.legend(loc="lower left")
    plt.show()


def normalize_output(y):
    # normalize angles using cosine and sine
    # ensures all values in [-1,1] and no large loss between -180 and 180 degrees
    y = y * np.pi / 180
    normalized_y = [[np.sin(row[0]),
                     np.cos(row[0]),
                     np.sin(row[1]),
                     np.cos(row[1])]
                    for row in y]
    return np.asarray(normalized_y)


def get_data(training=True):
    """
    Parse dataset for input and expected output values. If training, each dataset is split into training and validation
    data and outputs are normalized. A tensorflow Dataset is created for each file parsed, and the Datasets are
    concatenated. Concatenating tensorflow Datasets avoids the issue of sequences and batches spilling across different
    sets of data.

    :param training: (boolean) if training, each dataset is split into training and validation data. Outputs are
    normalized.
    :return:
        if training: train_ds (tensorflow Dataset), val_ds (tensorflow Dataset)
        else: test_ds (tensorflow Dataset), outputs (numpy Nx2 array)
    """
    datasets = {}
    if training:
        dir = '../data/training/'
    else:
        dir = '../data/testing/'
    for file_name in os.listdir(dir):
        x = np.genfromtxt(dir + file_name, delimiter='	', skip_header=13, usecols=(2, 3, 4, 5, 6, 7))
        y = np.genfromtxt(dir + file_name, delimiter='	', skip_header=13, usecols=(15, 16))
        datasets[file_name] = [x, y]

    if training:
        train_datasets = []
        val_datasets = []
        first = True
        for key in datasets.keys():
            data = datasets[key]
            x, y = data[0], data[1]
            y = normalize_output(y)

            # split data into 80% training, 20% validation
            len_train = (len(x) // 5) * 4
            x_train, x_val = np.split(x, [len_train])
            y_train, y_val = np.split(y, [len_train])

            if first:
                arr_x_train = x_train.copy()
                first = False
            else:
                arr_x_train = np.append(arr_x_train, x_train, axis=0)


            train_ds = make_dataset(x_train, y_train)
            val_ds = make_dataset(x_val, y_val)

            train_datasets.append(train_ds)
            val_datasets.append(val_ds)

        train_ds = train_datasets[0]
        for ds in train_datasets[1:]:
            train_ds = train_ds.concatenate(ds)

        # adapt input normalization layer using training data
        preprocessing_layer.adapt(arr_x_train)

        val_ds = val_datasets[0]
        for ds in val_datasets[1:]:
            val_ds = val_ds.concatenate(ds)

        return train_ds, val_ds
    else:
        test_datasets = []
        first = True
        for key in datasets.keys():
            data = datasets[key]
            x, y = data[0], data[1]
            test_ds = make_dataset(x)
            test_datasets.append(test_ds)

            if first:
                outputs = y.copy()
                first = False
            else:
                outputs = np.append(outputs, y, axis=0)
        test_ds = test_datasets[0]
        for ds in test_datasets[1:]:
            test_ds = test_ds.concatenate(ds)
        return test_ds, outputs


def get_sim_data(sim_id, training=True):
    """
    :param sim_id: (string)
    :param training: (boolean) if true, splits dataset into 2% testing, 20% validation, 78% training
    :return:
        if training: train_ds (Dataset), val_ds (Dataset),
                     test_ds (Dataset), y_test (numpy Nx2 array)
        else: test_ds (Dataset), y (numpy Nx2 array)
    """
    dir = '../data/sims/sim' + sim_id
    files = os.listdir(dir)
    acc = np.genfromtxt(os.path.join(dir, files[2]), delimiter=',', skip_header=1)
    gyr = np.genfromtxt(os.path.join(dir, files[5]), delimiter=',', skip_header=1)
    euler = np.genfromtxt(os.path.join(dir, files[3]), delimiter=',', skip_header=1, usecols=(1, 2))
    euler[:, [1, 0]] = euler[:, [0, 1]]  # swap columns so roll comes before pitch

    acc = ned2enu(acc)
    gyr = ned2enu(gyr)

    x = np.column_stack((acc, gyr))
    y = euler

    if training:
        len_val = len(x) // 5
        len_test = len(x) // 50
        len_train = len(x) - (len_val + len_test)

        x_train, x_val, x_test = np.split(x, [len_train, len_train + len_val])
        y_train, y_val, y_test = np.split(y, [len_train, len_train + len_val])
        y_train = normalize_output(y_train)
        y_val = normalize_output(y_val)

        preprocessing_layer.adapt(x_train)

        train_ds = make_dataset(x_train, y_train)
        val_ds = make_dataset(x_val, y_val)
        test_ds = make_dataset(x_test)

        return train_ds, val_ds, test_ds, y_test
    else:
        test_ds = make_dataset(x)
        return test_ds, y


def make_dataset(x, y=None):
    if y is not None:
        y = y[seq_length - 1:]
    ds = tf.keras.preprocessing.timeseries_dataset_from_array(
        data=x,
        targets=y,
        sequence_length=seq_length,
        batch_size=128)
    return ds


def ned2enu(data):
    data[:, [1, 0]] = data[:, [0, 1]]  # swap x and y
    data[:, 2] = - data[:, 2]  # negate z
    return data


def save(outputs, ref, filepath):
    """
    Function to write neural network outputs to file.
    :param outputs: Nx2 numpy array with roll and pitch angles from model output
    :param ref: Nx2 numpy array with roll and pitch reference angles
    :param filepath: location to save data to
    :return: None
    """
    ref = ref[seq_length - 1:]
    angles = np.column_stack((outputs, ref))
    np.savetxt(filepath, angles, delimiter='	', header='Model Roll	Model Pitch 	Ref Roll 	Ref Pitch')


if __name__ == '__main__':
    using_sim = True
    if using_sim:
        train, val, inputs, ref = get_sim_data(sim_id='1')
    else:
        train, val = get_data(training=True)
        inputs, ref = get_data(training=False)

    tune_hyperparameters(train, val)

    # train_model(train, val, 'mymodel_sim1_tuned')
    #
    # outputs = test_model(inputs, 'mymodel_sim1_tuned')
    # graph_model_output(outputs, ref)

    pass
