import tensorflow as tf
import numpy as np
import time
from datetime import datetime

# Logging function
def generate_log_filename(prefix='log'):
    now = datetime.now()
    return f'{prefix}_{now.strftime("%Y-%m-%d_%H%M")}.txt'

# Create TensorFlow dataset
def create_dataset(features, targets, input_lengths, target_lengths, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices((
        {
            'inputs': features,
            'input_lengths': input_lengths
        },
        {
            'targets': targets,
            'target_lengths': target_lengths
        }
    ))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset
    
# Config
vocab_size = 28
batch_size = 8
num_epochs = 5

# Setup device
device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
print(f"Using device: {device}")

# Load your real preprocessed data
features = np.load('features.npy')
targets = np.load('targets.npy')
input_lengths = np.load('input_lengths.npy')
target_lengths = np.load('target_lengths.npy')

# Create tf.data.Dataset
train_dataset = create_dataset(features, targets, input_lengths, target_lengths, batch_size)

@tf.keras.utils.register_keras_serializable()
class AcousticModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.fc1 = tf.keras.layers.Dense(128, activation='relu')
        self.fc2 = tf.keras.layers.Dense(vocab_size + 1)

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

with tf.device(device):
    model = AcousticModel()
    optimizer = tf.keras.optimizers.Adam(1e-3)
    
def compute_ctc_loss(logits, labels, logit_lengths, label_lengths):
    label_sparse = tf.keras.backend.ctc_label_dense_to_sparse(labels, label_lengths)
    loss = tf.nn.ctc_loss(
        labels=label_sparse,
        logits=logits,
        label_length=label_lengths,
        logit_length=logit_lengths,
        logits_time_major=True,
        blank_index=vocab_size
    )
    return tf.reduce_mean(loss)
    
# Training step
@tf.function
def train_step(batch):
    batch_features = batch[0]['inputs']
    batch_input_lengths = batch[0]['input_lengths']
    batch_targets = batch[1]['targets']
    batch_target_lengths = batch[1]['target_lengths']

    with tf.GradientTape() as tape:
        logits = model(batch_features)  # [batch, T, vocab+1]
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        log_probs = tf.transpose(log_probs, [1, 0, 2])  # [T, batch, vocab+1]

        loss = compute_ctc_loss(log_probs, batch_targets, batch_input_lengths, batch_target_lengths)
        
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Logging
log_filename = generate_log_filename('train_log')
log_file = open(log_filename, 'w')

print('Starting training...')
total_start = time.perf_counter()

for epoch in range(num_epochs):
    epoch_start = time.perf_counter()
    epoch_loss = 0.0
    batch_idx = 0

    for batch in train_dataset:
        with tf.device(device):
            loss = train_step(batch)
            epoch_loss += loss.numpy()
            batch_idx += 1

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx}], Loss: {loss.numpy():.4f}")

    epoch_end = time.perf_counter()
    avg_epoch_loss = epoch_loss / batch_idx
    print(f"Epoch {epoch + 1} average loss: {avg_epoch_loss:.4f}, duration: {epoch_end - epoch_start:.2f} seconds")
    log_file.write(f"Epoch {epoch + 1}: Avg Loss: {avg_epoch_loss:.4f}, Duration: {epoch_end - epoch_start:.2f} sec\n")

total_time = time.perf_counter() - total_start
print(f"Total training time: {total_time:.2f} seconds")
log_file.write(f"Total training time: {total_time:.2f} seconds\n")
log_file.write(f"Average epoch duration: {total_time / num_epochs:.2f} seconds\n")
log_file.close()

model.save('ctc_model_tf_saved.keras')
print("Training complete. Model saved to ctc_model_tf_saved.keras")

