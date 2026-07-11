import logging
import os
import urllib.request
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

DATA_PATH = "data/input.txt"
URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


@dataclass
class ModelConfig:
    vocab_size: int  # data-dependent: only known once the corpus is loaded
    context_length: int = 8
    embedding_dim: int = 32


@dataclass
class TrainConfig:
    batch_size: int = 32
    max_iterations: int = 10_000
    eval_iterations: int = 200  # kept small, 200 is enough for mean loss estimation
    learning_rate: float = 1e-3
    eval_interval: int = field(init=False)

    def __post_init__(self):
        self.eval_interval = self.max_iterations // 10


logging.basicConfig(level=logging.INFO)
torch.manual_seed(1_337)

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)
logging.info(f"Using {device} device")


class BigramLanguageModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.context_length = config.context_length
        n_embed = config.embedding_dim
        self.token_embedding_table = nn.Embedding(config.vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(config.context_length, n_embed)
        self.sa_head = SelfAttentionHead(config.context_length, n_embed, n_embed)
        self.lm_head = nn.Linear(n_embed, config.vocab_size)

    def forward(self, inputs, targets=None):
        B, T = inputs.shape

        token_embedding = self.token_embedding_table(inputs)
        position_embedding = self.position_embedding_table(
            torch.arange(T, device=device)
        )
        x = token_embedding + position_embedding
        x = self.sa_head(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            loss = F.cross_entropy(logits, targets.view(B * T))
        return logits, loss

    def generate(self, inputs, max_new_tokens):
        for _ in range(max_new_tokens):
            cropped_inputs = inputs[:, -self.context_length :]
            logits, _ = self(cropped_inputs)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=1)
            next_input = torch.multinomial(probs, num_samples=1)
            inputs = torch.cat((inputs, next_input), dim=1)
        return inputs


class SelfAttentionHead(nn.Module):
    def __init__(self, context_length, embedding_dim, head_size):
        super().__init__()
        self.query = nn.Linear(embedding_dim, head_size, bias=False)
        self.key = nn.Linear(embedding_dim, head_size, bias=False)
        self.value = nn.Linear(embedding_dim, head_size, bias=False)
        self.register_buffer(
            "tril", torch.tril(torch.ones(context_length, context_length))
        )

    def forward(self, inputs):
        B, T, C = inputs.shape

        k = self.key(inputs)
        q = self.query(inputs)

        weights = q @ k.transpose(-2, -1) * C**-0.5  # unit variance normalization
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        weights = F.softmax(weights, dim=-1)
        v = self.value(inputs)
        out = weights @ v
        return out


def get_batch(data, context_length, batch_size):
    ix = torch.randint(len(data) - context_length, (batch_size,))
    x = torch.stack([data[i : i + context_length] for i in ix])
    y = torch.stack([data[i + 1 : i + context_length + 1] for i in ix])
    return x.to(device), y.to(device)


def get_raw_data():
    if not os.path.exists(DATA_PATH):
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        urllib.request.urlretrieve(URL, DATA_PATH)
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return f.read()


@torch.no_grad()
def estimate_loss(model, train_data, val_data, train_config: TrainConfig, model_config: ModelConfig):
    runs = [("train", train_data), ("val", val_data)]
    out = {}
    model.eval()
    for split, d in runs:
        losses = torch.zeros(train_config.eval_iterations)
        for k in range(train_config.eval_iterations):
            x, y = get_batch(d, model_config.context_length, train_config.batch_size)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


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
    model_config = ModelConfig(vocab_size=len(vocab))
    train_config = TrainConfig()
    model = BigramLanguageModel(model_config).to(device)
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
    opt = optim.AdamW(model.parameters(), lr=train_config.learning_rate)
    # Training loop
    train_data, val_data = train_test_split(data, train_ratio=0.9)
    for i in range(train_config.max_iterations):
        if i % train_config.eval_interval == 0:
            losses = estimate_loss(model, train_data, val_data, train_config, model_config)
            logging.info(
                f"step {i}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )

        x, y = get_batch(train_data, model_config.context_length, train_config.batch_size)
        logits, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

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
