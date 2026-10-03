from __future__ import annotations

import logging
import os
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import override

import torch
from torch import nn, optim
from torch.nn import functional as F

DATA_PATH = "data/input.txt"
URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


@dataclass
class ModelConfig:
    vocab_size: int  # data-dependent: only known once the corpus is loaded
    context_length: int = 256
    embedding_dimension: int = 384  # 6 heads of head_size 64 each
    number_of_heads: int = 6
    number_of_layers: int = 6
    dropout_rate: float = 0.2


@dataclass
class TrainConfig:
    batch_size: int = 64
    max_iterations: int = 5_000
    eval_iterations: int = 500  # kept small, 200 is enough for mean loss estimation
    learning_rate: float = 3e-4
    eval_interval: int = field(init=False)

    def __post_init__(self):
        self.eval_interval = self.max_iterations // 10


logging.basicConfig(level=logging.INFO)
_ = torch.manual_seed(1_337)

device = (
    torch.accelerator.current_accelerator()
    if torch.accelerator.is_available()
    else "cpu"
)
logger = logging.getLogger(__name__)
logger.info(f"Using {device} device")


class BigramLanguageModel(nn.Module):
    context_length: int
    token_embedding_table: nn.Embedding
    position_embedding_table: nn.Embedding
    blocks: nn.Sequential
    final_layer_norm: nn.LayerNorm
    lm_head: nn.Linear

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.context_length = config.context_length
        embedding_dimension = config.embedding_dimension
        self.token_embedding_table = nn.Embedding(
            config.vocab_size, embedding_dimension
        )
        self.position_embedding_table = nn.Embedding(
            config.context_length, embedding_dimension
        )
        self.blocks = nn.Sequential(
            *[
                TransformerBlock(
                    self.context_length,
                    embedding_dimension,
                    config.number_of_heads,
                    config.dropout_rate,
                )
                for _ in range(config.number_of_layers)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(embedding_dimension)
        self.lm_head = nn.Linear(embedding_dimension, config.vocab_size)

    @override
    def forward(
        self, inputs: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, sequence_length = inputs.shape

        token_embedding = self.token_embedding_table(inputs)
        position_embedding = self.position_embedding_table(
            torch.arange(sequence_length, device=device)
        )
        x = token_embedding + position_embedding
        x = self.blocks(x)
        x = self.final_layer_norm(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            batch, sequence_length, vocab_size = logits.shape
            logits = logits.view(batch * sequence_length, vocab_size)
            loss = F.cross_entropy(logits, targets.view(batch * sequence_length))
        return logits, loss

    def generate(self, inputs: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        for _ in range(max_new_tokens):
            cropped_inputs = inputs[:, -self.context_length :]
            logits, _ = self(cropped_inputs)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=1)
            next_input = torch.multinomial(probs, num_samples=1)
            inputs = torch.cat((inputs, next_input), dim=1)
        return inputs


class TransformerBlock(nn.Module):
    mh_sa: MultiHeadedSelfAttention
    ffwd: FeedForward
    layer_norm_1: nn.LayerNorm
    layer_norm_2: nn.LayerNorm

    def __init__(
        self,
        context_length: int,
        embedding_dimension: int,
        number_of_heads: int,
        dropout_rate: float,
    ):
        super().__init__()
        self.mh_sa = MultiHeadedSelfAttention(
            context_length,
            embedding_dimension,
            number_of_heads,
            dropout_rate,
        )
        self.ffwd = FeedForward(embedding_dimension, dropout_rate)
        # feature normalization to get N(0,1) initialization
        self.layer_norm_1 = nn.LayerNorm(embedding_dimension)
        self.layer_norm_2 = nn.LayerNorm(embedding_dimension)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mh_sa(self.layer_norm_1(x))  # residual connections
        x = x + self.ffwd(self.layer_norm_2(x))  # residual connections
        return x


class FeedForward(nn.Module):
    net: nn.Sequential

    def __init__(self, embedding_dimension: int, dropout_rate: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(
                embedding_dimension, 4 * embedding_dimension
            ),  # 4x is the heuristic from Attention paper
            nn.ReLU(),
            nn.Linear(
                4 * embedding_dimension, embedding_dimension
            ),  # projection to get output
            nn.Dropout(dropout_rate),
        )

    @override
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class MultiHeadedSelfAttention(nn.Module):
    num_heads: int
    head_size: int
    qkv: nn.Linear
    tril: torch.Tensor  # pyright: ignore[reportUninitializedInstanceVariable]  # set via register_buffer
    dropout: nn.Dropout
    projection: nn.Linear

    def __init__(
        self,
        context_length: int,
        embedding_dimension: int,
        num_heads: int,
        dropout_rate: float,
    ):
        super().__init__()
        assert embedding_dimension % num_heads == 0, (
            f"number of heads {num_heads} does not divide embedding dimension {embedding_dimension}"
        )
        self.num_heads = num_heads
        self.head_size = embedding_dimension // num_heads
        self.qkv = nn.Linear(
            embedding_dimension, 3 * embedding_dimension, bias=False
        )  # Q, K, V matrices combined into one, and across num_heads of Attention
        self.register_buffer(
            "tril", torch.tril(torch.ones(context_length, context_length))
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.projection = nn.Linear(embedding_dimension, embedding_dimension)

    @override
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_length, sequence_length, embedding_dimension = inputs.shape  # (B, T, d)

        q, k, v = torch.split(
            self.qkv(inputs), embedding_dimension, dim=-1
        )  # 3 x (B, T, d)
        q = q.view(
            batch_length, sequence_length, self.num_heads, self.head_size
        ).transpose(1, 2)  # (B, H, T, Q)
        assert q.shape == (
            batch_length,
            self.num_heads,
            sequence_length,
            self.head_size,
        ), (
            f"Incorrect view for Q {q.shape}, expect {(batch_length, self.num_heads, sequence_length, self.head_size)}"
        )
        k_transpose = (
            k.view(batch_length, sequence_length, self.num_heads, self.head_size)
            .transpose(1, 2)
            .transpose(-2, -1)
        )  # (B, H, Q, T)
        assert k_transpose.shape == (
            batch_length,
            self.num_heads,
            self.head_size,
            sequence_length,
        ), (
            f"Incorrect view for K^T {k_transpose.shape}, expect {(batch_length, self.num_heads, self.head_size, sequence_length)}"
        )
        weights = (
            q @ k_transpose * embedding_dimension**-0.5
        )  # bug, should be head_size**-0.5 # (B, H, T, T)
        assert weights.shape == (
            batch_length,
            self.num_heads,
            sequence_length,
            sequence_length,
        ), (
            f"Incorrect shape for weights {weights.shape}, expected {(batch_length, self.num_heads, sequence_length, sequence_length)}"
        )
        weights = weights.masked_fill(
            self.tril[:sequence_length, :sequence_length] == 0, float("-inf")
        )
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)  # attention dropout

        v = v.view(
            batch_length, sequence_length, self.num_heads, self.head_size
        ).transpose(1, 2)  # (B, H, T, Q)
        assert v.shape == (
            batch_length,
            self.num_heads,
            sequence_length,
            self.head_size,
        ), (
            f"Incorrect view for V {v.shape}, expect {(batch_length, self.num_heads, sequence_length, self.head_size)}"
        )
        out = (weights @ v).transpose(1, 2)
        # (B, H, T, T) @ (B, H, T, Q) = (B, H, T, Q) -> (B, T, H, Q)
        return self.dropout(  # residual dropout
            self.projection(
                out.contiguous().view(
                    batch_length, sequence_length, embedding_dimension
                )  # (B, T, d)
            )
        )


def get_batch(
    data: torch.Tensor, context_length: int, batch_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(data) - context_length, (batch_size,))
    x = torch.stack([data[i : i + context_length] for i in ix])
    y = torch.stack([data[i + 1 : i + context_length + 1] for i in ix])
    return x.to(device), y.to(device)


def get_raw_data() -> str:
    if not os.path.exists(DATA_PATH):
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        _ = urllib.request.urlretrieve(URL, DATA_PATH)
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return f.read()


@torch.no_grad()
def estimate_loss(
    model: BigramLanguageModel,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    train_config: TrainConfig,
    model_config: ModelConfig,
):
    runs = [("train", train_data), ("val", val_data)]
    out = {}
    _ = model.eval()
    for split, d in runs:
        losses = torch.zeros(train_config.eval_iterations)
        for k in range(train_config.eval_iterations):
            x, y = get_batch(d, model_config.context_length, train_config.batch_size)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    _ = model.train()
    return out


def get_data(raw_data: str):
    vocab = build_vocab(raw_data)
    return torch.tensor(
        encoder(string_to_int_mapping(vocab))(raw_data), dtype=torch.long
    ), vocab


def build_vocab(data: str) -> list[str]:
    vocab = sorted(set(data))
    return vocab


def train_test_split(data: torch.Tensor, train_ratio: float = 0.9):
    train_size = int(train_ratio * len(data))
    return data[:train_size], data[train_size:]


def string_to_int_mapping(vocab: list[str]) -> dict[str, int]:
    return {ch: i for i, ch in enumerate(vocab)}


def int_to_string_mapping(vocab: list[str]) -> dict[int, str]:
    return {i: ch for i, ch in enumerate(vocab)}


def encoder(mapping: dict[str, int]) -> Callable[[str], list[int]]:
    return lambda string: [mapping[ch] for ch in string]


def decoder(mapping: dict[int, str]) -> Callable[[list[int]], str]:
    return lambda lst: "".join(mapping[i] for i in lst)


def main():
    data, vocab = get_data(get_raw_data())
    logger.info(f"Loaded {len(data)} characters")
    logger.info(f"Vocab of size {len(vocab)}")
    logger.info(f"Loaded data {data.shape} {data[:100]}")
    model_config = ModelConfig(vocab_size=len(vocab))
    train_config = TrainConfig()
    model = BigramLanguageModel(model_config).to(device)
    decoder_fn = decoder(int_to_string_mapping(vocab))
    logger.info(
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
            losses = estimate_loss(
                model, train_data, val_data, train_config, model_config
            )
            logger.info(
                f"step {i}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )

        x, y = get_batch(
            train_data, model_config.context_length, train_config.batch_size
        )
        # logits, loss
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        _ = opt.step()

    logger.info(
        f"After training {
            decoder_fn(
                model.generate(
                    torch.zeros((1, 1), dtype=torch.long).to(device),
                    max_new_tokens=1_000,
                )[0].tolist()
            )
        }"
    )


if __name__ == "__main__":
    main()
