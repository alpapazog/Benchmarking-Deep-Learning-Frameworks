import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import time
from datetime import datetime
from models import SentimentLSTM
# ✅ Logging function
def generate_log_filename(prefix='log'):
    now = datetime.now()
    return f'{prefix}_{now.strftime("%Y-%m-%d_%H%M")}.txt'

# ✅ Setup
vocab_size = 10000
maxlen = 200
embed_dim = 100
hidden_dim = 128
output_dim = 2
batch_size = 64
num_epochs = 5

# ✅ Enable XLA (optional but boosts speed in some cases)
tf.config.optimizer.set_jit(True)

# ✅ Device
device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
print('Using device:', device)

# ✅ Load and pre-tokenize dataset outside TensorFlow graph (efficient)
print('Loading and pre-tokenizing IMDB dataset (CPU)...')
(train_data, _), info = tfds.load(
    'imdb_reviews',
    split=['train', 'test'],
    as_supervised=True,
    with_info=True
)

train_texts = [text.numpy().decode('utf-8') for text, _ in train_data]
train_labels = np.array([label.numpy() for _, label in train_data])

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size, oov_token='<OOV>')
tokenizer.fit_on_texts(train_texts)

sequences = tokenizer.texts_to_sequences(train_texts)
sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=maxlen, padding='post')

# ✅ Create TensorFlow dataset from pre-tokenized data
train_dataset = tf.data.Dataset.from_tensor_slices((sequences, train_labels)).shuffle(len(sequences)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# ✅ Instantiate model, optimizer, and loss
with tf.device(device):
    model = SentimentLSTM(vocab_size, embed_dim, hidden_dim, output_dim)
    optimizer = tf.keras.optimizers.Adam(1e-3)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# ✅ Compiled training step
@tf.function
def train_step(batch_inputs, batch_labels):
    with tf.GradientTape() as tape:
        logits = model(batch_inputs, training=True)
        loss = loss_fn(batch_labels, logits)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

# ✅ Logging
log_filename = generate_log_filename('train_log')
log_file = open(log_filename, 'w')

# ✅ Training loop
print('Training model...')
total_start = time.perf_counter()

for epoch in range(1, num_epochs + 1):
    epoch_start = time.perf_counter()
    epoch_loss = 0.0
    batch_idx = 0

    for batch_inputs, batch_labels in train_dataset:
        loss = train_step(batch_inputs, batch_labels)
        epoch_loss += loss.numpy()
        batch_idx += 1

        if batch_idx % 100 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.numpy():.4f}')

    epoch_end = time.perf_counter()
    epoch_duration = epoch_end - epoch_start
    print(f'Epoch {epoch} avg loss: {epoch_loss / batch_idx:.4f} duration: {epoch_duration:.2f} seconds')
    log_file.write(f'Epoch {epoch}: {epoch_duration:.2f} sec\n')

total_end = time.perf_counter()
total_duration = total_end - total_start
print(f'Total training time: {total_duration:.2f} seconds')
log_file.write(f'Total training time: {total_duration:.2f} seconds\n')
log_file.write(f'Average epoch duration: {total_duration / num_epochs:.2f} seconds\n')
log_file.close()

# Save model
model.save('sentiment_lstm_tf_saved.keras')
print('Training complete. Model saved to sentiment_lstm_tf_saved')
print(f'Training log saved to {log_filename}')

