import os
import torchaudio
import torch
import numpy as np
from tqdm import tqdm

# --------------------------
# Parameters
# --------------------------
DATA_DIR = r"train-clean-5/LibriSpeech/train-clean-5"
OUTPUT_FILE = "librispeech_20min_dataset_model.pt"
target_duration_seconds = 400 * 60  # 200 minutes
sample_rate = 16000  # LibriSpeech default

# --------------------------
# Tokenizer (basic A-Z + space -> 1-27)
# --------------------------
def tokenize(transcript):
    return [ord(c) - ord('A') + 1 if c.isalpha() else 27 for c in transcript.upper() if c.isalpha() or c == ' ']

# --------------------------
# Load dataset & accumulate until 20 minutes
# --------------------------
samples = []
total_duration_sec = 0.0

print("✅ Scanning LibriSpeech...")

for root, _, files in os.walk(DATA_DIR):
    for file in files:
        if file.endswith(".txt"):
            with open(os.path.join(root, file), 'r') as f:
                for line in f:
                    parts = line.strip().split(maxsplit=1)
                    if len(parts) != 2:
                        continue
                    utt_id, transcript = parts
                    audio_path = os.path.join(root, utt_id + ".flac")
                    if os.path.exists(audio_path):
                        waveform, sr = torchaudio.load(audio_path)
                        duration = waveform.size(1) / sr
                        if total_duration_sec + duration > target_duration_seconds:
                            continue  # skip if exceeding 20 minutes
                        targets = tokenize(transcript)
                        samples.append((waveform, sr, targets))
                        total_duration_sec += duration
    if total_duration_sec >= target_duration_seconds:
        break

print(f"✅ Collected {len(samples)} utterances totaling approx {total_duration_sec / 60:.2f} minutes.")

# --------------------------
# Feature extraction and padding
# --------------------------
features_list, targets_list, input_lengths_list, target_lengths_list = [], [], [], []

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=sample_rate,
    n_mels=80
)

print("✅ Extracting features...")

for waveform, sr, targets in tqdm(samples):
    log_mel = mel_transform(waveform)
    log_mel = torch.log(log_mel + 1e-9).squeeze(0).transpose(0, 1)  # [time, n_mels]
    features_list.append(log_mel)
    targets_list.append(torch.tensor(targets, dtype=torch.long))
    input_lengths_list.append(torch.tensor(log_mel.size(0), dtype=torch.long))
    target_lengths_list.append(torch.tensor(len(targets), dtype=torch.long))

features_padded = torch.nn.utils.rnn.pad_sequence(features_list, batch_first=True, padding_value=0.0)
targets_padded = torch.nn.utils.rnn.pad_sequence(targets_list, batch_first=True, padding_value=0)
input_lengths_tensor = torch.stack(input_lengths_list)
target_lengths_tensor = torch.stack(target_lengths_list)

print(f"✅ Final shapes:\nFeatures: {features_padded.shape}\nTargets: {targets_padded.shape}")

# --------------------------
# Save using TorchScript extra_files (safe for C++)
# --------------------------

# Dummy scripted module
class DummyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

module = torch.jit.script(DummyModule())

# Save buffers
extra_files = {
    "features": features_padded.numpy().astype(np.float32).tobytes(),
    "targets": targets_padded.numpy().astype(np.int64).tobytes(),
    "input_lengths": input_lengths_tensor.numpy().astype(np.int64).tobytes(),
    "target_lengths": target_lengths_tensor.numpy().astype(np.int64).tobytes(),
}

torch.jit.save(module, OUTPUT_FILE, _extra_files=extra_files)

print(f"✅ Saved dataset to {OUTPUT_FILE} (TorchScript with extra_files)")
