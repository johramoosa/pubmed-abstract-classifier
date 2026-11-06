import os
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path

import torch
from fastapi import FastAPI, Request
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictStr,
    field_validator,
)

from .inference import predict_raw_abstract
from .models import PositionCNN


class PredictionRequest(BaseModel):
    """A complete abstract supplied as ordered, raw sentences."""

    model_config = ConfigDict(extra="forbid")

    sentences: list[StrictStr] = Field(min_length=1)

    @field_validator("sentences")
    @classmethod
    def reject_blank_sentences(cls, sentences):
        if any(not sentence.strip() for sentence in sentences):
            raise ValueError("Every sentence must be non-empty.")
        return sentences


def create_app(model_path=None):
    """Create an API whose model is loaded during application startup."""

    # Allow an explicit path, an environment setting, or the project default.
    if model_path is None:
        default_path = (
            Path(__file__).resolve().parent.parent
            / "artifacts"
            / "abstract_classifier_v1.pt"
        )
        model_path = os.environ.get(
            "ABSTRACT_CLASSIFIER_MODEL_PATH",
            str(default_path),
        )

    model_path = Path(model_path)

    @asynccontextmanager
    async def lifespan(app):
        # Startup: load the exported model once per application process.
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Exported model not found: {model_path}"
            )

        bundle = torch.load(
            model_path,
            map_location="cpu",
            weights_only=True,
        )

        if bundle["format_version"] != 1:
            raise ValueError("Unsupported model bundle format.")

        if bundle["model_class"] != "PositionCNN":
            raise ValueError("The API requires a PositionCNN bundle.")

        vocab = bundle["vocab"]
        label_to_id = bundle["label_to_id"]
        config = bundle["config"]

        model = PositionCNN(
            vocab_size=len(vocab),
            embedding_dim=config["embedding_dim"],
            num_filters=config["num_filters"],
            num_classes=len(label_to_id),
            pad_id=bundle["pad_id"],
            kernel_size=config["kernel_size"],
            padding=config["padding"],
        )

        model.load_state_dict(bundle["model_state_dict"])
        model.eval()

        id_to_label = {
            label_id: label
            for label, label_id in label_to_id.items()
        }

        # Retain the loaded model and mappings for subsequent requests.
        app.state.predictor = partial(
            predict_raw_abstract,
            model=model,
            vocab=vocab,
            id_to_label=id_to_label,
        )

        yield

        # Shutdown: release the application's reference to the predictor.
        del app.state.predictor

    app = FastAPI(
        title="Biomedical Abstract Classifier",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health():
        """Startup completed and the prediction model is available."""
        return {"status": "ok", "model_loaded": True}

    @app.post("/predict")
    def predict(payload: PredictionRequest, request: Request):
        """Classify the ordered sentences of one complete abstract."""
        predictions = request.app.state.predictor(payload.sentences)

        return {
            "predictions": predictions.to_dict(orient="records"),
        }

    return app


app = create_app()
