"""CPU board encoder. The output is a deterministic H-step chain of categorical emissions."""

import torch
from torch import nn

BINS = 9  # attacks 0..6, 7+, death


class Residual(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.ReLU(),
                                    nn.Conv2d(width, width, 3, padding=1))

    def forward(self, x):
        return torch.relu(x + self.layers(x))


class Predictor(nn.Module):
    def __init__(self, horizon=8, width=32, blocks=4, context_size=3, board=True):
        super().__init__()
        self.horizon = horizon
        self.use_board = board
        if board:
            yy, xx = torch.meshgrid(torch.linspace(-1, 1, 20), torch.linspace(-1, 1, 10), indexing="ij")
            self.register_buffer("coordinates", torch.stack((xx, yy))[None])
            self.encoder = nn.Sequential(nn.Conv2d(3, width, 3, padding=1), nn.ReLU(),
                                         *[Residual(width) for _ in range(blocks)],
                                         nn.Conv2d(width, 8, 1), nn.ReLU(), nn.Flatten())
        self.head = nn.Sequential(nn.Linear((1600 if board else 0) + context_size, 256), nn.ReLU(),
                                  nn.Linear(256, horizon * BINS))

    def forward(self, board, context):
        if self.use_board:
            coordinates = self.coordinates.expand(len(board), -1, -1, -1)
            encoded = self.encoder(torch.cat((board, coordinates), dim=1))
            context = torch.cat((encoded, context), dim=1)
        return self.head(context).reshape(-1, self.horizon, BINS)
