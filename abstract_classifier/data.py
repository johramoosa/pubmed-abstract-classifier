from pathlib import Path

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from .preprocessing import tokenize


def load_abstract_sentences(file_path):
    """Read PubMed RCT text into one row per labelled sentence."""
    rows = []
    abstract_id = None
    sentence_number = 0

    with Path(file_path).open(encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.rstrip("\r\n")

            # A blank line marks the end of an abstract.
            if not line.strip():
                abstract_id = None
                continue

            # An ID line starts a new abstract.
            if line.startswith("###"):
                abstract_id = line[3:].strip()
                sentence_number = 0

                if not abstract_id:
                    raise ValueError(
                        f"Missing abstract ID at line {line_number}"
                    )
                continue

            if abstract_id is None:
                raise ValueError(
                    f"Sentence without an abstract ID at line {line_number}"
                )

            if "\t" not in line:
                raise ValueError(
                    f"Missing label/text separator at line {line_number}"
                )

            label, sentence = line.split("\t", maxsplit=1)

            rows.append({
                "abstract_id": abstract_id,
                "sentence": sentence,
                "label": label,
                "sentence_number": sentence_number,
            })

            sentence_number += 1

    return pd.DataFrame(
        rows,
        columns=["abstract_id", "sentence", "label", "sentence_number"],
    )


class AbstractSentenceDataset(Dataset):
    """Return token IDs and a label ID for each sentence."""

    def __init__(self, dataframe, vocab, label_to_id):
        self.sentences = dataframe["sentence"].tolist()
        self.labels = dataframe["label"].tolist()
        self.vocab = vocab
        self.label_to_id = label_to_id

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        token_ids = [
            self.vocab.get(token, self.vocab["<UNK>"])
            for token in tokenize(self.sentences[index])
        ]

        return (
            torch.tensor(token_ids, dtype=torch.long),
            torch.tensor(
                self.label_to_id[self.labels[index]],
                dtype=torch.long,
            ),
        )


class PositionDataset(Dataset):
    """Return token IDs, relative position, and a label ID."""

    def __init__(self, dataframe, vocab, label_to_id):
        self.sentences = dataframe["sentence"].tolist()
        self.positions = dataframe["relative_position"].tolist()
        self.labels = dataframe["label"].tolist()
        self.vocab = vocab
        self.label_to_id = label_to_id

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        token_ids = [
            self.vocab.get(token, self.vocab["<UNK>"])
            for token in tokenize(self.sentences[index])
        ]

        return (
            torch.tensor(token_ids, dtype=torch.long),
            torch.tensor(self.positions[index], dtype=torch.float32),
            torch.tensor(
                self.label_to_id[self.labels[index]],
                dtype=torch.long,
            ),
        )


def collate_batch(batch, pad_id):
    """Pad a batch of (token_ids, label_id) examples."""
    token_sequences, labels = zip(*batch)

    padded_tokens = pad_sequence(
        token_sequences,
        batch_first=True,
        padding_value=pad_id,
    )

    return padded_tokens, torch.stack(labels)


def collate_position_batch(batch, pad_id):
    """Pad a batch of (token_ids, relative_position, label_id) examples."""
    token_sequences, positions, labels = zip(*batch)

    padded_tokens = pad_sequence(
        token_sequences,
        batch_first=True,
        padding_value=pad_id,
    )

    return (
        padded_tokens,
        torch.stack(positions),
        torch.stack(labels),
    )


def make_seeded_train_loader(existing_loader, seed):
    """Recreate a training loader with a reproducible shuffle generator."""
    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset=existing_loader.dataset,
        batch_size=existing_loader.batch_size,
        shuffle=True,
        collate_fn=existing_loader.collate_fn,
        drop_last=existing_loader.drop_last,
        generator=generator,
        num_workers=0,
    )
