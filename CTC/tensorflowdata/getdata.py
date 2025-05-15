import os
import numpy as np
import soundfile as sf
import tensorflow as tf

# Config
data_dir = '/media/alex/00C4FC10C4FC09A4/DevUMD/605/GroupProject/CTC/cpp/data'
n_mels = 80
sample_rate = 16000
vocab = list("_'ABCDEFGHIJKLMNOPQRSTUVWXYZ ")

# Character-level tokenizer (CTC-style)
char_to_id = {c: i + 1 for i, c in enumerate(vocab[1:])}
char_to_id[vocab[0]] = 0  # blank at index 0

# Feature extractor using standard TF APIs (safe and portable)
def extract_features(wav_file):
    audio, sr = sf.read(wav_file)
    if sr != sample_rate:
        raise ValueError(f"Expected {sample_rate}Hz but got {sr}")
    audio = tf.convert_to_tensor(audio, dtype=tf.float32)
    spectrogram = tf.signal.stft(audio, frame_length=400, frame_step=160, fft_length=512)
    magnitude = tf.abs(spectrogram)
    mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(n_mels, 257, sample_rate, 80.0, 7600.0)
    mel_spectrogram = tf.tensordot(magnitude, mel_weight_matrix, 1)
    log_mel_spectrogram = tf.math.log(tf.maximum(mel_spectrogram, 1e-10))
    return log_mel_spectrogram.numpy()

# Label encoder
def encode_transcript(transcript):
    transcript = transcript.upper()
    return np.array([char_to_id.get(c, char_to_id[" "]) for c in transcript if c in char_to_id], dtype=np.int32)

# Extract data
features_list = []
targets_list = []
input_lengths_list = []
target_lengths_list = []
file_count = 0

for root, dirs, files in os.walk(data_dir):
    if file_count > 2400:
        break;
    for file in files:
        if file_count > 2400:
            break;
        if file.endswith('.txt'):
            with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split(' ', 1)
                    if len(parts) != 2:
                        continue
                    file_id, transcript = parts
                    flac_path = os.path.join(root, file_id + '.flac')
                    if not os.path.exists(flac_path):
                        continue
                    try:
                        feature = extract_features(flac_path)
                        label = encode_transcript(transcript)
                        features_list.append(feature)
                        targets_list.append(label)
                        input_lengths_list.append(feature.shape[0])
                        target_lengths_list.append(len(label))
                        print(f"Processed {flac_path}")
                        file_count += 1
                        if file_count > 2400:
                            break;
                    except Exception as e:
                        print(f"Failed for {flac_path}: {e}")

# Padding
from tensorflow.keras.preprocessing.sequence import pad_sequences

features_padded = tf.keras.preprocessing.sequence.pad_sequences(features_list, padding='post', dtype='float32')
targets_padded = pad_sequences(targets_list, padding='post', dtype='int32')
input_lengths = np.array(input_lengths_list, dtype=np.int32)
target_lengths = np.array(target_lengths_list, dtype=np.int32)

print(f"Features shape: {features_padded.shape}")
print(f"Targets shape: {targets_padded.shape}")
print(f"Input lengths: {input_lengths.shape}")
print(f"Target lengths: {target_lengths.shape}")

# Save to .npy
np.save('data/features.npy', features_padded)
np.save('data/targets.npy', targets_padded)
np.save('data/input_lengths.npy', input_lengths)
np.save('data/target_lengths.npy', target_lengths)

print(f"Extraction complete: Total audio duration")

