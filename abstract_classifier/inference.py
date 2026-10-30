"""Predict section labels using trained text-only or position-aware models."""

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence

from .preprocessing import tokenize, prepare_raw_sentence


def _validate_sentences(sentences):
    """Validate an ordered collection of non-empty sentences."""
    if isinstance(sentences, str):
        raise TypeError(
            "Pass an ordered list of sentences, not a whole paragraph."
        )

    sentences = list(sentences)

    if not sentences:
        raise ValueError("Provide at least one sentence.")

    if any(
        not isinstance(sentence, str) or not sentence.strip()
        for sentence in sentences
    ):
        raise ValueError("Every sentence must be a non-empty string.")

    return sentences


def predict_section(sentence, *, model, vocab, id_to_label):
    """Classify one sentence using a text-only model."""
    if not isinstance(sentence, str) or not sentence.strip():
        raise ValueError("Please provide a non-empty sentence.")

    token_ids = [
        vocab.get(token, vocab["<UNK>"])
        for token in tokenize(sentence)
    ]

    device = next(model.parameters()).device
    inputs = torch.tensor(
        token_ids,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)

    model.eval()

    with torch.no_grad():
        predicted_id = model(inputs).argmax(dim=1).item()

    return id_to_label[predicted_id]


def predict_abstract(sentences, *, model, vocab, id_to_label):
    """Classify a complete, ordered abstract using a position-aware model.

    Sentences should already have dataset-style punctuation spacing.
    """
    sentences = _validate_sentences(sentences)

    encoded_sentences = [
        torch.tensor(
            [
                vocab.get(token, vocab["<UNK>"])
                for token in tokenize(sentence)
            ],
            dtype=torch.long,
        )
        for sentence in sentences
    ]

    token_ids = pad_sequence(
        encoded_sentences,
        batch_first=True,
        padding_value=vocab["<PAD>"],
    )

    total_sentences = len(sentences)
    positions = (
        torch.arange(total_sentences, dtype=torch.float32)
        / max(total_sentences - 1, 1)
    )

    device = next(model.parameters()).device
    token_ids = token_ids.to(device)
    positions = positions.to(device)

    model.eval()

    with torch.no_grad():
        logits = model(token_ids, positions)
        predicted_ids = logits.argmax(dim=1).cpu().tolist()

    return pd.DataFrame({
        "sentence_number": range(total_sentences),
        "relative_position": positions.cpu().tolist(),
        "sentence": sentences,
        "predicted_label": [
            id_to_label[index] for index in predicted_ids
        ],
    })


def predict_raw_abstract(sentences, *, model, vocab, id_to_label):
    """Prepare raw sentences with the limited adapter, then classify them.

    Input must be a complete abstract supplied as an ordered sentence list.
    This function does not split paragraphs into sentences.
    """
    original_sentences = _validate_sentences(sentences)

    prepared_sentences = [
        prepare_raw_sentence(sentence)
        for sentence in original_sentences
    ]

    predictions = predict_abstract(
        prepared_sentences,
        model=model,
        vocab=vocab,
        id_to_label=id_to_label,
    )

    predictions = predictions.rename(
        columns={"sentence": "prepared_sentence"}
    )
    predictions.insert(2, "sentence", original_sentences)

    return predictions
