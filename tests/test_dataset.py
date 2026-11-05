import pandas as pd
import torch

from abstract_classifier.data import PositionDataset


def test_position_dataset_encodes_example():
    """Encode words, unknown tokens, position, and label correctly."""
    vocab = {
        "<PAD>": 0,
        "<UNK>": 1,
        "patients": 2,
        "improved": 3,
    }

    label_to_id = {
        "METHODS": 0,
        "RESULTS": 1,
    }

    examples = pd.DataFrame({
        "sentence": ["Patients improved unexpectedly"],
        "relative_position": [0.5],
        "label": ["RESULTS"],
    })

    dataset = PositionDataset(examples, vocab, label_to_id)

    assert len(dataset) == 1

    token_ids, position, label_id = dataset[0]

    # Lowercase "Patients"; map unseen "unexpectedly" to <UNK>.
    expected_tokens = torch.tensor([2, 3, 1], dtype=torch.long)
    expected_position = torch.tensor(0.5, dtype=torch.float32)
    expected_label = torch.tensor(1, dtype=torch.long)

    torch.testing.assert_close(
        token_ids, expected_tokens, rtol=0, atol=0
    )
    torch.testing.assert_close(
        position, expected_position, rtol=0, atol=0
    )
    torch.testing.assert_close(
        label_id, expected_label, rtol=0, atol=0
    )
