import pytest
import torch
from fastapi.testclient import TestClient

from abstract_classifier.api import create_app
from abstract_classifier.models import PositionCNN


@pytest.fixture
def client(tmp_path):
    """Start the real API using a tiny temporary model bundle."""
    vocab = {
        "<PAD>": 0,
        "<UNK>": 1,
        "patients": 2,
        "improved": 3,
        ".": 4,
    }

    label_to_id = {
        "BACKGROUND": 0,
        "OBJECTIVE": 1,
        "METHODS": 2,
        "RESULTS": 3,
        "CONCLUSIONS": 4,
    }

    config = {
        "embedding_dim": 8,
        "num_filters": 4,
        "kernel_size": 3,
        "padding": 1,
    }

    torch.manual_seed(42)

    model = PositionCNN(
        vocab_size=len(vocab),
        num_classes=len(label_to_id),
        pad_id=vocab["<PAD>"],
        **config,
    )

    bundle_path = tmp_path / "test_model.pt"

    torch.save(
        {
            "format_version": 1,
            "model_class": "PositionCNN",
            "model_state_dict": model.state_dict(),
            "vocab": vocab,
            "label_to_id": label_to_id,
            "pad_id": vocab["<PAD>"],
            "config": config,
        },
        bundle_path,
    )

    app = create_app(model_path=bundle_path)

    # Start the application before the test; shut it down afterward.
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_loaded": True,
    }


def test_predict_returns_ordered_results(client):
    sentences = [
        "Patients improved.",
        "Patients improved.",
        "Patients improved.",
    ]

    response = client.post(
        "/predict",
        json={"sentences": sentences},
    )

    assert response.status_code == 200

    predictions = response.json()["predictions"]

    assert len(predictions) == len(sentences)
    assert [row["sentence_number"] for row in predictions] == [0, 1, 2]
    assert [row["relative_position"] for row in predictions] == [
        0.0, 0.5, 1.0
    ]
    assert [row["sentence"] for row in predictions] == sentences
    assert [row["prepared_sentence"] for row in predictions] == [
        "Patients improved .",
        "Patients improved .",
        "Patients improved .",
    ]

    valid_labels = {
        "BACKGROUND", "OBJECTIVE", "METHODS",
        "RESULTS", "CONCLUSIONS",
    }

    assert all(
        row["predicted_label"] in valid_labels
        for row in predictions
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"sentences": []},
        {"sentences": ["   "]},
        {"sentences": "Patients improved."},
    ],
)
def test_predict_rejects_invalid_input(client, payload):
    response = client.post("/predict", json=payload)

    assert response.status_code == 422
