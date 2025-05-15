import tensorflow as tf
import numpy as np

# ✅ Decorate your custom model
@tf.keras.utils.register_keras_serializable()
class AcousticModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.fc1 = tf.keras.layers.Dense(128, activation='relu')
        self.fc2 = tf.keras.layers.Dense(29)  # vocab_size + 1 (CTC blank)

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x
        
# Load your trained model
model = tf.keras.models.load_model('ctc_model_tf_saved.keras', compile=False)

# Load one example (e.g., first sample)
sample_input = features[0:1]  # [1, 1726, 80]
sample_input_length = input_lengths[0:1]  # [1]

# Forward pass
logits = model(sample_input)  # [1, 1726, vocab+1]
log_probs = tf.nn.log_softmax(logits, axis=-1)
log_probs = tf.transpose(log_probs, [1, 0, 2])  # [T, batch, vocab+1]

# Decode using greedy decoder (you can also use beam search)
decoded, log_prob = tf.nn.ctc_greedy_decoder(
    inputs=log_probs,
    sequence_length=sample_input_length
)

# Extract decoded sequence (sparse tensor to dense)
decoded_dense = tf.sparse.to_dense(decoded[0], default_value=-1)

print(f"Decoded sequence (token ids): {decoded_dense.numpy()}")

