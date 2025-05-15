import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import time
from datetime import datetime
from models import TransformerClassifier

# Logging function
def generate_log_filename(prefix='log'):
    now = datetime.now()
    return f'{prefix}_{now.strftime("%Y-%m-%d_%H%M")}.txt'

# Setup
vocab_size = 10000
maxlen = 200
embed_dim = 512
num_heads = 8
ff_hidden_dim = 2048
num_classes = 2
batch_size = 64
num_epochs = 5

# Enable XLA
tf.config.optimizer.set_jit(True)
device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
print('Using device:', device)

# Load IMDB and pre-tokenize
print('Loading IMDB...')
(train_data, _), _ = tfds.load('imdb_reviews', split=['train', 'test'], as_supervised=True, with_info=True)
texts = [text.numpy().decode('utf-8') for text, _ in train_data]
labels = np.array([label.numpy() for _, label in train_data])

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size, oov_token='<OOV>')
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=maxlen, padding='post')

dataset = tf.data.Dataset.from_tensor_slices((sequences, labels)).shuffle(len(sequences)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Model
with tf.device(device):
    model = TransformerClassifier(vocab_size, embed_dim, maxlen, num_heads, ff_hidden_dim, num_classes)
    optimizer = tf.keras.optimizers.Adam(1e-4)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

@tf.function
def train_step(batch_inputs, batch_labels):
    with tf.GradientTape() as tape:
        logits = model(batch_inputs, training=True)
        loss = loss_fn(batch_labels, logits)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

# Logging
log_filename = generate_log_filename('train_log')
log_file = open(log_filename, 'w')

print('Training model...')
total_start = time.perf_counter()

for epoch in range(1, num_epochs + 1):
    epoch_start = time.perf_counter()
    epoch_loss = 0.0
    batch_idx = 0

    for batch_inputs, batch_labels in dataset:
        loss = train_step(batch_inputs, batch_labels)
        epoch_loss += loss.numpy()
        batch_idx += 1

        if batch_idx % 100 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.numpy():.4f}')

    epoch_end = time.perf_counter()
    print(f'Epoch {epoch} avg loss: {epoch_loss / batch_idx:.4f} duration: {epoch_end - epoch_start:.2f} sec')
    log_file.write(f'Epoch {epoch}: {epoch_end - epoch_start:.2f} sec\n')

total_end = time.perf_counter()
print(f'Total training time: {total_end - total_start:.2f} seconds')
log_file.write(f'Total training time: {total_end - total_start:.2f} seconds\n')
log_file.write(f'Average epoch duration: {(total_end - total_start) / num_epochs:.2f} seconds\n')
log_file.close()

model.save('transformer_imdb_tf_saved.keras')
print('Training complete. Model saved to transformer_imdb_tf_saved.keras')
print(f'Training log saved to {log_filename}')

