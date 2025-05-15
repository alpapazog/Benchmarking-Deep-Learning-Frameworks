import tensorflow as tf
import numpy as np
import time
from datetime import datetime

# Setup device
device = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"
print("Device:", device)

# Logging
def generate_log_filename(prefix='log'):
    now = datetime.now()
    return f"{prefix}_{now.strftime('%Y-%m-%d_%H%M')}.txt"
    
@tf.function
def train_step(x_batch, y_batch):
    with tf.GradientTape() as tape:
        logits = model(x_batch, training=True)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_batch, logits)
        loss = tf.reduce_mean(loss)

    grads = tape.gradient(loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

log_filename = generate_log_filename('train_log')
log_file = open(log_filename, 'w')

# Load and preprocess MNIST dataset
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype(np.float32) / 255.0
x_train = (x_train - 0.1307) / 0.3081
x_train = np.expand_dims(x_train, axis=-1)  # (N, 28, 28, 1)

# Load all data to GPU tensor
with tf.device(device):
    x_train_tensor = tf.convert_to_tensor(x_train)
    y_train_tensor = tf.convert_to_tensor(y_train)

# Define the model (same structure as PyTorch model)
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(10, 5, activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Conv2D(20, 5, activation='relu'),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax'),
])

# Compile
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# Settings
batch_size = 64
num_epochs = 5
num_batches = x_train_tensor.shape[0] // batch_size

# Custom training loop (to match your PyTorch style)
start_time = time.time()

for epoch in range(1, num_epochs + 1):
    epoch_start = time.time()
    indices = tf.random.shuffle(tf.range(x_train_tensor.shape[0]))
    for batch_idx in range(num_batches):
        idx = indices[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        x_batch = tf.gather(x_train_tensor, idx)
        y_batch = tf.gather(y_train_tensor, idx)

        # Use compiled train step
        loss = train_step(x_batch, y_batch)

        if batch_idx % 100 == 0:
            print(f'Epoch {epoch} | Batch {batch_idx} | Loss: {loss.numpy():.4f}')

    epoch_end = time.time()
    print(f'Epoch {epoch} duration: {epoch_end - epoch_start:.2f} seconds')
    log_file.write(f'Epoch {epoch}: {epoch_end - epoch_start:.2f} sec\n')

total_duration = time.time() - start_time
print(f'Total training time: {total_duration:.2f} seconds')
log_file.write(f'Total training time: {total_duration:.2f} seconds\n')
log_file.write(f'Average epoch duration: {total_duration / num_epochs:.2f} seconds\n')
log_file.close()

# Save model
model.save('mnist_cnn_tf_preloaded.h5')
print('Model saved to mnist_cnn_tf_preloaded.h5')

