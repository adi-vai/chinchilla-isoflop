import logging
import os
import urllib.request

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

DATA_PATH = "data/input.txt"
URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

CONTEXT_LENGTH = 8
BATCH_SIZE = 32

logging.basicConfig(level=logging.INFO)
torch.manual_seed(1_337)

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)
logging.info(f"Using {device} device")


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, inputs, targets=None):

        logits = self.token_embedding_table(inputs)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            loss = F.cross_entropy(logits, targets.view(B * T))
        return logits, loss

    def generate(self, inputs, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(inputs)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=1)
            next_input = torch.multinomial(probs, num_samples=1)
            inputs = torch.cat((inputs, next_input), dim=1)
        return inputs


def get_batch(data):
    ix = torch.randint(len(data) - CONTEXT_LENGTH, (BATCH_SIZE,))
    x = torch.stack([data[i : i + CONTEXT_LENGTH] for i in ix])
    y = torch.stack([data[i + 1 : i + CONTEXT_LENGTH + 1] for i in ix])
    return x.to(device), y.to(device)


def get_raw_data():
    if not os.path.exists(DATA_PATH):
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        urllib.request.urlretrieve(URL, DATA_PATH)
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return f.read()


def get_data(raw_data):
    vocab = build_vocab(raw_data)
    return torch.tensor(
        encoder(string_to_int_mapping(vocab))(raw_data), dtype=torch.long
    ), vocab


def build_vocab(data):
    vocab = sorted(list(set(data)))
    return vocab


def train_test_split(data, train_ratio=0.9):
    train_size = int(train_ratio * len(data))
    return data[:train_size], data[train_size:]


def string_to_int_mapping(vocab):
    return {ch: i for i, ch in enumerate(vocab)}


def int_to_string_mapping(vocab):
    return {i: ch for i, ch in enumerate(vocab)}


def encoder(mapping):
    return lambda string: [mapping[ch] for ch in string]


def decoder(mapping):
    return lambda lst: "".join(mapping[i] for i in lst)


def main():
    data, vocab = get_data(get_raw_data())
    logging.info(f"Loaded {len(data)} characters")
    logging.info(f"Vocab of size {len(vocab)}")
    logging.info(f"Loaded data {data.shape} {data[:100]}")
    x, y = get_batch(data)
    model = BigramLanguageModel(len(vocab)).to(device)
    decoder_fn = decoder(int_to_string_mapping(vocab))
    logging.info(
        f"Before training {
            decoder_fn(
                model.generate(
                    torch.zeros((1, 1), dtype=torch.long).to(device), max_new_tokens=100
                )[0].tolist()
            )
        }"
    )
    train_steps = 10_000
    opt = optim.AdamW(model.parameters(), lr=1e-3)
    for i in range(train_steps):
        logits, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if i % 1_000 == 0:
            logging.info(f"Current loss is {loss}")
    logging.info(
        f"After training {
            decoder_fn(
                model.generate(
                    torch.zeros((1, 1), dtype=torch.long).to(device), max_new_tokens=100
                )[0].tolist()
            )
        }"
    )
    return


if __name__ == "__main__":
    main()
