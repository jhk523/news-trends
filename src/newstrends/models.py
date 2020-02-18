from torch import nn


class EmbeddingModel(nn.Module):
    def __init__(self, num_pieces, embedding_dim=8):
        super().__init__()
        self.embedding = nn.Embedding(num_pieces, embedding_dim)
