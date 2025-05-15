import torch
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import pickle

MAX_LEN = 200
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"

tokenizer = get_tokenizer("basic_english")

def yield_tokens(data_iter):
    for label, text in data_iter:
        yield tokenizer(text)

# Build vocabulary
train_iter_raw = list(IMDB(split='train'))
test_iter_raw = list(IMDB(split='test'))
vocab = build_vocab_from_iterator((tokenizer(text) for label, text in train_iter_raw), specials=[PAD_TOKEN, UNK_TOKEN])
vocab.set_default_index(vocab[UNK_TOKEN])

# Save vocab
with open("./data/vocab.pkl", "wb") as f:
    pickle.dump(vocab, f)

# Encode and pad
# def process(data_iter):
#     inputs, labels = [], []
#     for label, text in data_iter:
#         token_ids = vocab(tokenizer(text))[:MAX_LEN]
#         token_ids += [vocab[PAD_TOKEN]] * (MAX_LEN - len(token_ids))
#         inputs.append(torch.tensor(token_ids, dtype=torch.int64))
#         labels.append(torch.tensor(1 if label == "pos" else 0, dtype=torch.int64))
#     return torch.stack(inputs), torch.tensor(labels)
def process(data_iter):
    inputs, labels = [], []
    for label, text in data_iter:
        token_ids = vocab(tokenizer(text))[:MAX_LEN]
        token_ids += [vocab[PAD_TOKEN]] * (MAX_LEN - len(token_ids))
        inputs.append(torch.tensor(token_ids, dtype=torch.int64))

        # Fix label interpretation (1=neg, 2=pos in torchtext)
        label_id = 1 if label == 2 else 0
        labels.append(torch.tensor(label_id, dtype=torch.int64))
    return torch.stack(inputs), torch.tensor(labels)



X_train, Y_train = process(train_iter_raw)
X_test, Y_test = process(test_iter_raw)
print("Train positive:", (Y_train == 1).sum().item())
print("Train negative:", (Y_train == 0).sum().item())

# Save in compatible format
torch.save({"inputs": X_train, "labels": Y_train}, "./data/dataset.pt")

print("Train positive:", (Y_test == 1).sum().item())
print("Train negative:", (Y_test == 0).sum().item())

# Save in compatible format
torch.save({"inputs": X_test, "labels": Y_test}, "./data/dataset_test.pt")

print("Saved ./data/dataset_test.pt with shape:", X_test.shape, Y_test.shape)

print(f"Vocab size: {len(vocab)}")