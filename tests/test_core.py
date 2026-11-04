import pandas as pd
import torch

from abstract_classifier.models import PositionCNN
from abstract_classifier.checkpointing import save_checkpoint
from abstract_classifier.preprocessing import add_position_feature


def test_position_cnn_checkpoint_round_trip(tmp_path):
    """Saving and loading must preserve the model's output."""
    model_config = {
        "vocab_size": 12,
        "embedding_dim": 8,
        "num_filters": 4,
        "num_classes": 5,
        "pad_id": 0,
    }

    original_model = PositionCNN(**model_config)
    original_model.eval()

    token_ids = torch.tensor([
        [2, 3, 4, 0],
        [5, 6, 7, 8],
    ], dtype=torch.long)

    positions = torch.tensor([0.0, 1.0], dtype=torch.float32)

    with torch.no_grad():
        expected_scores = original_model(token_ids, positions)

    checkpoint_path = tmp_path / "model.pt"

    save_checkpoint(
        {
            "model_state_dict": original_model.state_dict(),
            "config": model_config,
        },
        checkpoint_path,
    )

    assert checkpoint_path.is_file(), "Checkpoint was not created."

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    loaded_model = PositionCNN(**checkpoint["config"])
    loaded_model.load_state_dict(checkpoint["model_state_dict"])
    loaded_model.eval()

    with torch.no_grad():
        actual_scores = loaded_model(token_ids, positions)

    assert expected_scores.shape == (2, 5)
    assert torch.isfinite(expected_scores).all()

    torch.testing.assert_close(
        actual_scores,
        expected_scores,
        rtol=0,
        atol=0,
    )


def test_relative_position():
    """Check positions for one-, two-, and three-sentence abstracts."""
    examples = pd.DataFrame({
        "abstract_id": ["A", "B", "B", "C", "C", "C"],
        "sentence": [
            "Only sentence.",
            "First sentence.",
            "Last sentence.",
            "First sentence.",
            "Middle sentence.",
            "Last sentence.",
        ],
        "sentence_number": [0, 0, 1, 0, 1, 2],
    })

    result = add_position_feature(examples)

    assert result["total_sentences"].tolist() == [1, 2, 2, 3, 3, 3]

    expected_positions = torch.tensor(
        [0.0, 0.0, 1.0, 0.0, 0.5, 1.0],
        dtype=torch.float64,
    )

    actual_positions = torch.tensor(
        result["relative_position"].to_numpy(),
        dtype=torch.float64,
    )

    torch.testing.assert_close(
        actual_positions,
        expected_positions,
        rtol=0,
        atol=0,
    )
