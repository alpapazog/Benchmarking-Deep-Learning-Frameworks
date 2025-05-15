import tensorflow as tf
from keras.saving import register_keras_serializable
import math

def positional_encoding(seq_len, embed_dim):
    position = tf.cast(tf.range(seq_len)[:, tf.newaxis], tf.float32)
    div_term = tf.exp(tf.cast(tf.range(0, embed_dim, 2), tf.float32) * -(math.log(10000.0) / embed_dim))
    pe_even = tf.sin(position * div_term)
    pe_odd = tf.cos(position * div_term)
    pe = tf.concat([pe_even, pe_odd], axis=-1)
    return pe  # [seq_len, embed_dim]

@register_keras_serializable()
class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_hidden_dim):
        super().__init__()
        self.self_attn = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.norm1 = tf.keras.layers.LayerNormalization()
        self.norm2 = tf.keras.layers.LayerNormalization()
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_hidden_dim, activation='relu'),
            tf.keras.layers.Dense(embed_dim),
        ])

    def call(self, x):
        attn_output = self.self_attn(x, x)
        x = self.norm1(x + attn_output)
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        return x

@register_keras_serializable()
class TransformerClassifier(tf.keras.Model):
    def __init__(self, vocab_size, embed_dim, seq_len, num_heads, ff_hidden_dim, num_classes):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_dim)
        self.encoder = TransformerEncoder(embed_dim, num_heads, ff_hidden_dim)
        self.classifier = tf.keras.layers.Dense(num_classes)
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.pe = positional_encoding(seq_len, embed_dim)[tf.newaxis, :, :]

    def call(self, input_ids):
        x = self.embedding(input_ids)
        x = x + self.pe
        x = self.encoder(x)
        x = tf.reduce_mean(x, axis=1)
        logits = self.classifier(x)
        return logits

