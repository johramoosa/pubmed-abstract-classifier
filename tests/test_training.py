import math

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from abstract_classifier.models import PositionCNN
from abstract_classifier.training import train_position_epoch


def test_training_updates_parameters():
    """A training step must produce finite loss and update parameters."""
    torch.manual_seed(42)

    model = PositionCNN(
        vocab_size=12,
        embedding_dim=8,
        num_filters=4,
        num_classes=5,
        pad_id=0,
    )

    # Two invented examples, processed together in one batch.
    tokens = torch.tensor([
        [2, 3, 4],
        [5, 6, 0],
    ], dtype=torch.long)

    positions = torch.tensor([0.0, 1.0], dtype=torch.float32)
    labels = torch.tensor([1, 4], dtype=torch.long)

    loader = DataLoader(
        TensorDataset(tokens, positions, labels),
        batch_size=2,
        shuffle=False,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    # Keep independent copies of the parameters before training.
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }

    loss, accuracy = train_position_epoch(
        model,
        loader,
        loss_fn,
        optimizer,
    )

    assert math.isfinite(loss), "Training loss is not finite."
    assert 0.0 <= accuracy <= 1.0

    assert all(
        torch.isfinite(parameter).all().item()
        for parameter in model.parameters()
    ), "Training produced non-finite parameters."

    assert any(
        not torch.equal(before[name], parameter.detach())
        for name, parameter in model.named_parameters()
    ), "No parameters changed during training."
