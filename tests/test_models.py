import torch

from abstract_classifier.models import PositionCNN


def test_position_cnn_ignores_trailing_padding():
    """Extra trailing padding must not change a sentence's scores."""
    model = PositionCNN(
        vocab_size=12,
        embedding_dim=8,
        num_filters=4,
        num_classes=5,
        pad_id=0,
    )
    model.eval()

    # Same sentence, with different amounts of padding.
    original_tokens = torch.tensor([[2, 3, 4]], dtype=torch.long)
    padded_tokens = torch.tensor([[2, 3, 4, 0, 0]], dtype=torch.long)

    position = torch.tensor([0.5], dtype=torch.float32)

    with torch.no_grad():
        original_scores = model(original_tokens, position)
        padded_scores = model(padded_tokens, position)

    torch.testing.assert_close(
        padded_scores,
        original_scores,
        rtol=1e-5,
        atol=1e-6,
    )
