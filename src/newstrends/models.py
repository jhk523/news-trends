import typing

import torch
from torch import nn


class EmbeddingModel(nn.Module):
    def __init__(self, num_pieces, num_classes, embedding_dim=8):
        super().__init__()
        self.embedding = nn.Embedding(num_pieces, embedding_dim)
        self.linear = nn.Linear(embedding_dim, num_classes)

    # noinspection PyShadowingBuiltins
    def __call__(self, *input, **kwargs) -> typing.Any:
        return super().__call__(*input, **kwargs)

    def forward(self, x):
        out = torch.matmul(x, self.embedding.weight)
        return self.linear(out)
