import re


def tokenize(sentence):
    """Match the tokenizer used to train the saved model."""
    return sentence.lower().split()


def prepare_raw_sentence(sentence):
    """Apply limited punctuation spacing to an already-separated sentence."""
    if not isinstance(sentence, str) or not sentence.strip():
        raise ValueError("Provide a non-empty sentence.")

    # Preserve numeric commas, such as the comma in 1,000.
    sentence = re.sub(
        r"(?<!\d),|,(?!\d)|[;:!?()\[\]{}]",
        r" \g<0> ",
        sentence,
    )

    # Separate a final period; preserve internal periods.
    sentence = re.sub(
        r'\.(?=[\s"\'”’)\]}]*$)',
        " . ",
        sentence,
    )

    return " ".join(sentence.split())


def add_position_feature(df):
    """Add abstract length and relative position using zero-based sentence numbers."""
    result = df.copy()

    result["total_sentences"] = (
        result.groupby("abstract_id")["sentence"].transform("size")
    )

    # Avoid division by zero for a one-sentence abstract.
    denominator = (result["total_sentences"] - 1).clip(lower=1)

    result["relative_position"] = (
        result["sentence_number"] / denominator
    )

    return result
