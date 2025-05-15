import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import time
from datetime import datetime
from models import TransformerClassifier  # ✅ Always import your model first (registered)

# Logging function
def generate_log_filename(prefix='log'):
    now = datetime.now()
    return f'{prefix}_{now.strftime("%Y-%m-%d_%H%M")}.txt'

# Config (must match training)
vocab_size = 10000
maxlen = 200
embed_dim = 512
num_heads = 8
ff_hidden_dim = 2048
num_classes = 2
batch_size = 64

# Setup device
device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
print('Using device:', device)

# Load IMDB test data and pre-tokenize (CPU side)
print('Loading and pre-tokenizing IMDB test dataset...')
(_, test_data), _ = tfds.load('imdb_reviews', split=['train', 'test'], as_supervised=True, with_info=True)
test_texts = [text.numpy().decode('utf-8') for text, _ in test_data]
test_labels = np.array([label.numpy() for _, label in test_data])

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size, oov_token='<OOV>')
tokenizer.fit_on_texts(test_texts)
sequences = tokenizer.texts_to_sequences(test_texts)
sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=maxlen, padding='post')

test_dataset = tf.data.Dataset.from_tensor_slices((sequences, test_labels)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Load model
print('Loading model...')
with tf.device(device):
    model = tf.keras.models.load_model('transformer_imdb_tf_saved.keras')

# Reset GPU memory stats (optional, if using GPU)
gpu_devices = tf.config.list_physical_devices('GPU')
if gpu_devices:
    gpu_name = 'GPU:0'
    tf.config.experimental.reset_memory_stats(gpu_name)

# Warm-up
for batch_inputs, _ in test_dataset.take(1):
    _ = model(batch_inputs, training=False)

# Inference timing
start_time = time.perf_counter()
all_preds = []
for batch_inputs, _ in test_dataset:
    logits = model(batch_inputs, training=False)
    preds = tf.argmax(logits, axis=1)
    all_preds.append(preds.numpy())
end_time = time.perf_counter()

# Calculate stats
all_preds = np.concatenate(all_preds)
accuracy = np.mean(all_preds == test_labels) * 100.0
total_inference_time = end_time - start_time
avg_inference_time_per_sample = total_inference_time / len(test_labels)

# Get GPU memory usage (if applicable)
peak_used_mb = current_used_mb = 0
if gpu_devices:
    mem_info = tf.config.experimental.get_memory_info(gpu_name)
    peak_used_mb = mem_info['peak'] / (1024 ** 2)
    current_used_mb = mem_info['current'] / (1024 ** 2)

# Log and print
print(f'Inference Accuracy: {accuracy:.2f}%')
print(f'Total Inference Time: {total_inference_time:.4f} seconds')
print(f'Avg Inference Time per Sample: {avg_inference_time_per_sample:.6f} seconds')
print(f'Peak GPU memory used: {peak_used_mb:.2f} MB')
print(f'Current GPU memory after inference: {current_used_mb:.2f} MB')

log_filename = generate_log_filename('inference_log')
with open(log_filename, 'w') as log_file:
    log_file.write(f'Inference Accuracy: {accuracy:.2f}%\n')
    log_file.write(f'Total Inference Time: {total_inference_time:.4f} seconds\n')
    log_file.write(f'Avg Inference Time per Sample: {avg_inference_time_per_sample:.6f} seconds\n')
    log_file.write(f'Peak GPU memory used: {peak_used_mb:.2f} MB\n')
    log_file.write(f'Current GPU memory after inference: {current_used_mb:.2f} MB\n')

print(f'Inference log saved to {log_filename}')

