import torch

from abstract_classifier.data import collate_position_batch


def test_collate_position_batch():
    """Check padding and preserve each sentence's position and label."""

    # Use a distinct padding ID to verify that the supplied value is used.
    pad_id = 9

    batch = [
        (
            torch.tensor([2, 3, 4], dtype=torch.long),
            torch.tensor(0.0, dtype=torch.float32),
            torch.tensor(1, dtype=torch.long),
        ),
        (
            torch.tensor([5], dtype=torch.long),
            torch.tensor(1.0, dtype=torch.float32),
            torch.tensor(4, dtype=torch.long),
        ),
    ]

    tokens, positions, labels = collate_position_batch(
        batch,
        pad_id=pad_id,
    )

    expected_tokens = torch.tensor([
        [2, 3, 4],
        [5, 9, 9],
    ], dtype=torch.long)

    expected_positions = torch.tensor(
        [0.0, 1.0], dtype=torch.float32
    )

    expected_labels = torch.tensor(
        [1, 4], dtype=torch.long
    )

    # These comparisons check shape, dtype, and values.
    torch.testing.assert_close(
        tokens, expected_tokens, rtol=0, atol=0
    )
    torch.testing.assert_close(
        positions, expected_positions, rtol=0, atol=0
    )
    torch.testing.assert_close(
        labels, expected_labels, rtol=0, atol=0
    )
