import tensorflow as tf
from keras.saving import register_keras_serializable

@register_keras_serializable()
class SentimentLSTM(tf.keras.Model):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_dim)
        self.lstm = tf.keras.layers.LSTM(hidden_dim, return_sequences=True)
        self.fc = tf.keras.layers.Dense(output_dim)

    def call(self, x):
        x = self.embedding(x)
        x = self.lstm(x)
        last_hidden = x[:, -1, :]
        return self.fc(last_hidden)

