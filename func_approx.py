import logging

import numpy as np
import torch
from torch import nn, optim

logging.basicConfig(level=logging.INFO)

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)
logging.info(f"Using {device} device")


def generate_input(N: int):
    return torch.linspace(-2 * np.pi, 2 * np.pi, N).reshape(-1, 1)


def generate_target(x):
    return torch.sin(x).reshape(-1, 1)


# Model
class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.two_layer_mlp = nn.Sequential(
            nn.Linear(1, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        pred = self.two_layer_mlp(x)
        return pred


def build_model(device):
    model = Model().to(device)
    logging.info(f"Instantiated model {model}")
    return model


def build_loss():
    loss_function = nn.MSELoss()
    logging.info(f"Using loss function {loss_function}")
    return loss_function


def build_optimizer(model, lr=0.001):
    # Fixed LR, in reality we have a schedule
    opt = optim.Adam(model.parameters(), lr=lr)
    logging.info(f"Using optimizer {opt}")
    return opt


def train(model, input, target, loss_fn, opt, n_steps, log_every=1_000):
    history = []
    for i in range(n_steps):
        opt.zero_grad()
        output = model(input)
        loss = loss_fn(output, target)
        history.append(loss.item())
        if i % log_every == 0:
            # First loss is just MSE of function over interval
            logging.info(f"After {i} steps, loss is {loss}")
        loss.backward()
        opt.step()
    return history


def main():
    N = 1_000
    logging.info(f"Generating {N} samples of data")
    input = generate_input(N).to(device)
    logging.info(f"Input {input.shape}")
    target = generate_target(input).to(device)
    logging.info(f"Target {target.shape}")

    model = build_model(device)
    loss_function = build_loss()
    opt = build_optimizer(model)

    n_steps = 10_000
    logging.info(f"Running backprop for {n_steps} steps")
    history = train(model, input, target, loss_function, opt, n_steps)
    logging.info(f"Final loss after {n_steps} steps: {history[-1]}")


if __name__ == "__main__":
    main()
