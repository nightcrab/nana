"""Independent convolutional heads, dense fusion, and an autoregressive attack GRU."""

import torch
from torch import nn

BINS = 9  # attacks 0..6, 7+, death
START = BINS


class Residual(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.ReLU(),
                                    nn.Conv2d(width, width, 3, padding=1))

    def forward(self, x):
        return torch.relu(x + self.layers(x))


class ConvHead(nn.Sequential):
    def __init__(self, width, blocks):
        super().__init__(nn.Conv2d(3, width, 3, padding=1), nn.ReLU(),
                         *[Residual(width) for _ in range(blocks)],
                         nn.Conv2d(width, 1, 1), nn.ReLU(), nn.Flatten())


class Predictor(nn.Module):
    def __init__(self, horizon=8, width=4, blocks=2, context_size=3, board=True,
                 heads=32, hidden=128, fusion=256, embedding=16):
        super().__init__()
        self.horizon = horizon
        self.use_board = board
        if board:
            yy, xx = torch.meshgrid(torch.linspace(-1, 1, 20), torch.linspace(-1, 1, 10), indexing="ij")
            self.register_buffer("coordinates", torch.stack((xx, yy))[None])
        self.heads = nn.ModuleList([ConvHead(width, blocks) for _ in range(heads)] if board else [])
        self.fusion = nn.Sequential(nn.Linear(200 * len(self.heads) + context_size, fusion), nn.ReLU())
        self.initial = nn.Linear(fusion, hidden)
        self.embedding = nn.Embedding(BINS + 1, embedding)
        self.gru = nn.GRU(embedding + fusion, hidden, batch_first=True)
        self.output = nn.Linear(hidden, BINS)

    def encode(self, board, context):
        if self.use_board:
            x = torch.cat((board, self.coordinates.expand(len(board), -1, -1, -1)), dim=1)
            context = torch.cat([head(x) for head in self.heads] + [context], dim=1)
        return self.fusion(context)

    @staticmethod
    def absorbing(logits, previous):
        # Death is terminal, as in the recorded targets. No probability leaks into resurrection.
        terminal = torch.full_like(logits, -torch.inf)
        terminal[..., BINS - 1] = 0
        return torch.where((previous == BINS - 1)[..., None], terminal, logits)

    def decode(self, conditioning, targets):
        """Teacher-forced logits. Each replay must supply its own target prefix."""
        start = targets.new_full((len(targets), 1), START)
        previous = torch.cat((start, targets[:, :-1]), dim=1)
        inputs = torch.cat((self.embedding(previous),
                            conditioning[:, None].expand(-1, targets.shape[1], -1)), dim=-1)
        states, _ = self.gru(inputs, self.initial(conditioning)[None])
        return self.absorbing(self.output(states), previous)

    def forward(self, board, context, targets):
        return self.decode(self.encode(board, context), targets)

    @torch.no_grad()
    def trajectories(self, board, context, beam=32, topk=3):
        """Batched beam search; log probabilities are absolute, never top-k normalized.

        Beam selection is an offline coverage baseline. A finite beam approximates the
        best sequences; a CPU implementation can independently explore prefixes/samples.
        """
        if beam < topk or topk < 1:
            raise ValueError("beam must be >= topk >= 1")
        conditioning = self.encode(board, context)
        n = len(conditioning)
        hidden = self.initial(conditioning)[:, None]  # batch, beam, hidden
        sequences = torch.empty(n, 1, 0, dtype=torch.long, device=board.device)
        scores = conditioning.new_zeros(n, 1)
        previous = sequences.new_full((n, 1), START)
        for _ in range(self.horizon):
            k = scores.shape[1]
            inputs = torch.cat((self.embedding(previous), conditioning[:, None].expand(-1, k, -1)), -1)
            states, next_hidden = self.gru(inputs.reshape(n*k, 1, -1), hidden.reshape(1, n*k, -1))
            logits = self.absorbing(self.output(states[:, 0]).reshape(n, k, BINS), previous)
            extended = scores[..., None] + logits.log_softmax(-1)
            scores, indices = extended.flatten(1).topk(min(beam, k*BINS), dim=1)
            parents, previous = indices // BINS, indices % BINS
            sequences = torch.cat((sequences.gather(1, parents[..., None].expand(-1, -1, sequences.shape[-1])),
                                   previous[..., None]), -1)
            hidden = next_hidden[0].reshape(n, k, -1).gather(1, parents[..., None].expand(-1, -1, hidden.shape[-1]))
        return sequences[:, :topk], scores[:, :topk]
