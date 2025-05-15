import torch

dataset = torch.load("ctc_dataset.pt")

features = dataset['features']
targets = dataset['targets']
input_lengths = dataset['input_lengths']
target_lengths = dataset['target_lengths']

print(f"Features shape: {features.shape}")
print(f"Targets shape: {targets.shape}")
print(f"Input lengths shape: {input_lengths.shape}")
print(f"Target lengths shape: {target_lengths.shape}")

print("\nExample data check:")
idx = 0  # first example
print(f"First feature tensor shape: {features[idx].shape}")
print(f"First target sequence (numeric): {targets[idx]}")
print(f"First input length: {input_lengths[idx]}")
print(f"First target length: {target_lengths[idx]}")

# Optional: Decode numeric target back to text for verification
decoded_transcript = ''.join(
    [chr(t + ord('A') - 1) if 1 <= t <= 26 else ' ' for t in targets[idx] if t != 0]
)
print(f"Decoded transcript example: {decoded_transcript}")
