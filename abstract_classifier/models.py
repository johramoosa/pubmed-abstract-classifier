import torch
from torch import nn


class AbstractClassifier(nn.Module):
    """Classify sentences using the mean of their token embeddings."""

    def __init__(self, vocab_size, embedding_dim, num_classes, pad_id):
        super().__init__()
        self.pad_id = pad_id

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_id,
        )

        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, token_ids):
        embeddings = self.embedding(token_ids)

        # Include real tokens only when calculating the mean.
        mask = (token_ids != self.pad_id).unsqueeze(-1)

        summed = (embeddings * mask).sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1)
        sentence_vectors = summed / lengths

        return self.classifier(sentence_vectors)


class CNNAbstractClassifier(nn.Module):
    """Classify sentences using convolution and masked max pooling."""

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        num_filters,
        num_classes,
        pad_id,
        kernel_size=3,
        padding=1,
    ):
        super().__init__()
        self.pad_id = pad_id

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_id,
        )

        self.conv = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=num_filters,
            kernel_size=kernel_size,
            padding=padding,
        )

        self.activation = nn.ReLU()
        self.classifier = nn.Linear(num_filters, num_classes)

    def forward(self, token_ids):
        embeddings = self.embedding(token_ids).transpose(1, 2)
        features = self.activation(self.conv(embeddings))

        # Exclude padded positions from max pooling.
        mask = (token_ids != self.pad_id).unsqueeze(1)
        features = features.masked_fill(~mask, float("-inf"))

        text_features = features.max(dim=2).values
        return self.classifier(text_features)


class PositionCNN(nn.Module):
    """Classify sentences using CNN text features and relative position."""

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        num_filters,
        num_classes,
        pad_id,
        kernel_size=3,
        padding=1,
    ):
        super().__init__()
        self.pad_id = pad_id

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_id,
        )

        self.conv = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=num_filters,
            kernel_size=kernel_size,
            padding=padding,
        )

        self.activation = nn.ReLU()
        self.classifier = nn.Linear(num_filters + 1, num_classes)

    def forward(self, token_ids, positions):
        embeddings = self.embedding(token_ids).transpose(1, 2)
        features = self.activation(self.conv(embeddings))

        mask = (token_ids != self.pad_id).unsqueeze(1)
        features = features.masked_fill(~mask, float("-inf"))

        text_features = features.max(dim=2).values

        combined = torch.cat(
            [text_features, positions.unsqueeze(1)],
            dim=1,
        )

        return self.classifier(combined)
