import tensorflow as tf
import numpy as np
import time
from datetime import datetime

# Setup device
device = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"
print("Device:", device)

# Load the model
with tf.device(device):
    model = tf.keras.models.load_model('mnist_cnn_tf_preloaded.h5')

# Load test dataset
(x_test, y_test), _ = tf.keras.datasets.mnist.load_data()
x_test = x_test.astype(np.float32) / 255.0
x_test = (x_test - 0.1307) / 0.3081
x_test = np.expand_dims(x_test, axis=-1)  # (N, 28, 28, 1)

batch_size = 64

# Logging setup
now = datetime.now()
log_filename = f'inference_log_{now.strftime("%Y-%m-%d_%H%M")}.txt'
log_file = open(log_filename, 'w')

# Check if GPU available
gpu_devices = tf.config.list_physical_devices('GPU')
if gpu_devices:
    gpu_name = 'GPU:0'  # Use this directly instead of parsing from device name
    tf.config.experimental.reset_memory_stats(gpu_name)

# Warm-up (important for accurate timings)
with tf.device(device):
    _ = model.predict(x_test[:batch_size], batch_size=batch_size, verbose=0)

# Start timing
start_time = time.perf_counter()

with tf.device(device):
    preds = model.predict(x_test, batch_size=batch_size, verbose=0)

end_time = time.perf_counter()

if gpu_devices:
    mem_info = tf.config.experimental.get_memory_info(gpu_name)
    peak_used_mb = mem_info['peak'] / (1024 ** 2)
    current_used_mb = mem_info['current'] / (1024 ** 2)
    print(f'Peak GPU memory used: {peak_used_mb:.2f} MB')
    print(f'Current GPU memory after inference: {current_used_mb:.2f} MB')



# Compute accuracy
pred_labels = np.argmax(preds, axis=1)
accuracy = np.mean(pred_labels == y_test) * 100.0

# Results
print(f'Inference Accuracy: {accuracy:.2f}%')
print(f'Total Inference Time: {end_time - start_time:.4f} seconds')
print(f'Avg Inference Time per Sample: {(end_time - start_time) / x_test.shape[0]:.6f} seconds')

# Logging
log_file.write(f'Inference Accuracy: {accuracy:.2f}%\n')
log_file.write(f'Total Inference Time: {end_time - start_time:.4f} seconds\n')
log_file.write(f'Avg Inference Time per Sample: {(end_time - start_time) / x_test.shape[0]:.6f} seconds\n')
log_file.write(f'Peak GPU memory used: {peak_used_mb:.2f} MB\n')
log_file.write(f'Current GPU memory after inference: {current_used_mb:.2f} MB\n')
log_file.close()

print(f'Log saved to {log_filename}')

