import typing

import torch
from torch import nn
from torch.nn import LayerNorm
from torch.nn.modules.transformer import TransformerEncoderLayer, TransformerEncoder


class SoftmaxClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes, embedding_dim=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, num_classes, bias=False)

    # noinspection PyShadowingBuiltins
    def __call__(self, *input, **kwargs) -> typing.Any:
        return super().__call__(*input, **kwargs)

    def forward(self, x):
        out = torch.matmul(x, self.embedding.weight)
        return self.linear(out)


def lookup_embedding(embedding, x, transpose_x=True, transpose_mask=True):
    if transpose_x and transpose_mask:
        x = x.transpose(0, 1)
        mask = x < 0
    elif transpose_x:
        mask = x < 0
        x = x.transpose(0, 1)
    elif transpose_mask:
        mask = x.transpose(0, 1) < 0
    else:
        mask = x < 0

    out = torch.clamp(x, min=0)
    out = embedding.weight.index_select(dim=0, index=out.view(-1))
    out = out.view(x.size(0), x.size(1), -1)
    return out, mask


class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes, embedding_dim=8,
                 cell_type='lstm', num_layers=1, dropout=0):
        super().__init__()
        hidden_size = 2 * embedding_dim
        rnn_dropout = 0 if num_layers == 1 else dropout

        assert cell_type in {'gru', 'lstm'}
        if cell_type == 'lstm':
            self.rnn = nn.LSTM(
                embedding_dim, hidden_size, num_layers, dropout=rnn_dropout)
        else:
            self.rnn = nn.GRU(
                embedding_dim, hidden_size, num_layers, dropout=rnn_dropout)

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dense = nn.Sequential(
            nn.Linear(hidden_size, embedding_dim),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Linear(embedding_dim, num_classes, bias=False))

    # noinspection PyShadowingBuiltins
    def __call__(self, *input, **kwargs) -> typing.Any:
        return super().__call__(*input, **kwargs)

    def forward(self, x):
        out, mask = lookup_embedding(self.embedding, x)
        out.masked_fill_(mask.unsqueeze(dim=2), value=0)
        out = self.rnn(out)[0][-1]
        return self.dense(out)


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes, embedding_dim, nhead=1,
                 num_encoder_layers=2):
        super().__init__()
        d_model = embedding_dim
        dim_feedforward = 2 * d_model

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward)
        encoder_norm = LayerNorm(d_model)
        self.encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm)
        self.linear = nn.Linear(d_model, num_classes)

    # noinspection PyShadowingBuiltins
    def __call__(self, *input, **kwargs) -> typing.Any:
        return super().__call__(*input, **kwargs)

    def forward(self, x):
        out, mask = lookup_embedding(self.embedding, x, transpose_mask=False)
        out = self.encoder(out, src_key_padding_mask=mask)
        return self.linear(out[0])
