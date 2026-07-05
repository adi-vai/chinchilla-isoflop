import logging

logging.basicConfig(level=logging.INFO)

import numpy as np
import torch
from torch import nn, optim

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)
print(f"Using {device} device")


def generate_input(N: int):
    return torch.linspace(-2 * np.pi, 2 * np.pi, N).reshape(-1, 1)


def generate_target(x):  # TODO: Fill in type
    return torch.sin(x).reshape(-1, 1)


def training_loop():
    return


# Model
class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Where does the 512 come from? Appropriate?
        self.two_layer_mlp = nn.Sequential(
            nn.Linear(1, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        pred = self.two_layer_mlp(x)
        return pred


def main():
    N = 1_000
    logging.info(f"Generating {N} samples of data")
    input = generate_input(N).to(device)
    logging.info(f"Input {input.shape}")
    target = generate_target(input).to(device)
    logging.info(f"Target {target.shape}")
    model = Model().to(device)
    logging.info(f"Instantiated model {model}")
    loss_function = nn.MSELoss()
    logging.info(f"Using loss function {loss_function}")
    n_steps = 10_000
    logging.info(f"Running backprop for {n_steps} steps")
    # Fixed LR, in reality we have a schedule
    opt = optim.Adam(model.parameters(), lr=0.001)
    logging.info(f"Using optimizer {opt}")
    for i in range(n_steps):
        opt.zero_grad()
        output = model(input)
        loss = loss_function(output, target)
        if i % 1_000 == 0:
            # First loss is just MSE of function over interval
            logging.info(f"After {i} steps, loss is {loss}")
        loss.backward()
        opt.step()


if __name__ == "__main__":
    main()
