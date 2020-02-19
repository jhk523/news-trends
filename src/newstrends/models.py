import typing

import torch
from torch import nn


class SoftmaxClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes, embedding_dim=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, num_classes)

    # noinspection PyShadowingBuiltins
    def __call__(self, *input, **kwargs) -> typing.Any:
        return super().__call__(*input, **kwargs)

    def forward(self, x):
        out = torch.matmul(x, self.embedding.weight)
        return self.linear(out)


class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes, embedding_dim=8, hidden_size=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_size)
        self.linear = nn.Linear(hidden_size, num_classes)

    # noinspection PyShadowingBuiltins
    def __call__(self, *input, **kwargs) -> typing.Any:
        return super().__call__(*input, **kwargs)

    def forward(self, x):
        x_t = x.transpose(0, 1)
        out = torch.clamp(x_t, min=0)
        out = self.embedding.weight.index_select(dim=0, index=out.view(-1))
        out = out.view(x_t.size(0), x_t.size(1), -1)
        out.masked_fill_((x_t < 0).unsqueeze(dim=2), value=0)
        out = self.gru(out)[-1].squeeze(dim=0)
        return self.linear(out)
