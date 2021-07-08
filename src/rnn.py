import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm


def get_batch(data, attitude, seq_length, batch_size):
    # the length of the input vector
    n = data.shape[0] - 1
    # randomly choose the starting indices for the examples in the training batch
    idx = np.random.choice(n - seq_length, batch_size)

    input_batch = [data[i:i + seq_length] for i in idx]
    output_batch = [attitude[i:i + seq_length] for i in idx]

    # x_batch, y_batch provide the true inputs and targets for network training
    x_batch = np.reshape(input_batch, [batch_size, seq_length, 6])
    y_batch = np.reshape(output_batch, [batch_size, seq_length, 2])
    return x_batch, y_batch


def simple_rnn(rnn_units):
    return tf.keras.layers.SimpleRNN(
        rnn_units,
        activation='relu',
        kernel_initializer='glorot_uniform',
        dropout=.4,
        return_sequences=True,
        stateful=True
    )


def build_model(rnn_units):
    model = tf.keras.Sequential([
        simple_rnn(rnn_units),
        tf.keras.layers.Dense(2)
    ])

    return model


def compute_loss(y_true, y_pred):
    loss = tf.keras.losses.mean_squared_error(y_true, y_pred)
    return loss


def train_step(x, y):
    # Use tf.GradientTape()
    with tf.GradientTape() as tape:
        y_hat = model(x)
        loss = compute_loss(y, y_hat)

    # Compute the gradients
    grads = tape.gradient(loss, model.trainable_variables)

    # Apply the gradients to the optimizer so it can update the model accordingly
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


def train_model(input, output, batch_size):
    history = []
    plt.ion()
    fig = plt.figure()
    if hasattr(tqdm, '_instances'): tqdm._instances.clear()  # clear if it exists

    for iter in tqdm(range(num_training_iterations)):

        # Grab a batch and propagate it through the network
        x_batch, y_batch = get_batch(input, output, seq_length, batch_size)
        loss = train_step(x_batch, y_batch)
        # Update the progress bar
        history.append(loss.numpy().mean())

        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.plot(history)
        plt.draw()
        plt.pause(2)

        # Update the model with the changed weights
        if iter % 100 == 0:
            model.save_weights(checkpoint_prefix)
    plt.ioff()
    plt.show()
    # Save the trained model and the weights
    model.save_weights(checkpoint_prefix)


def get_data():
    data = [[], []]
    for file_name in os.listdir('../data/training'):
        inFile = open('../data/training/' + file_name, 'r')
        for line in inFile:
            if line.startswith('//') or line.startswith('PacketCounter'):
                continue
            values = line.split('	')
            values = [float(x) for x in values]
            data[0].append(np.array([values[i] for i in range(2, 8)]))
            data[1].append(np.array([values[11], values[12]]))
    x = np.asarray(data[0])
    y = np.asarray(data[1])
    return x, y


# optimization parameters
num_training_iterations = 1000
batch_size = 50
seq_length = 250
learning_rate = 1e-4

# model parameters
rnn_units = 1500

# checkpoint location
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "my_ckpt")

model = build_model(rnn_units, batch_size)
optimizer = tf.keras.optimizers.Adam(learning_rate)

if __name__ == '__main__':
    x, y = get_data()
    train_model(x, y, batch_size)
    pass
