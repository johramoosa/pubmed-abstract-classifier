import torch


def _run_epoch(model, loader, loss_fn, optimizer=None):
    """Run an epoch with ordinary, unweighted mean-reduced loss.

    Accepts batches containing either:
        (tokens, labels)
        (tokens, positions, labels)

    Returns:
        mean_loss, accuracy, actual_labels, predictions
    """
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    actual_labels = []
    predictions = []

    with torch.set_grad_enabled(training):
        for batch in loader:
            # Everything except the final item is a model input.
            *inputs, labels = batch

            if training:
                optimizer.zero_grad()

            scores = model(*inputs)
            loss = loss_fn(scores, labels)

            if training:
                loss.backward()
                optimizer.step()

            predicted_labels = scores.argmax(dim=1)
            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size
            total_correct += (
                predicted_labels == labels
            ).sum().item()
            total_examples += batch_size

            actual_labels.extend(labels.detach().cpu().tolist())
            predictions.extend(predicted_labels.detach().cpu().tolist())

    if total_examples == 0:
        raise ValueError("The loader produced no examples.")

    return (
        total_loss / total_examples,
        total_correct / total_examples,
        actual_labels,
        predictions,
    )


def train_one_epoch(model, loader, loss_fn, optimizer):
    """Train a text-only model; return mean loss and accuracy."""
    loss, accuracy, _, _ = _run_epoch(
        model, loader, loss_fn, optimizer
    )
    return loss, accuracy


def evaluate(model, loader, loss_fn):
    """Evaluate a text-only model; return mean loss and accuracy."""
    loss, accuracy, _, _ = _run_epoch(
        model, loader, loss_fn
    )
    return loss, accuracy


def train_position_epoch(model, loader, loss_fn, optimizer):
    """Train a position-aware model; return mean loss and accuracy."""
    loss, accuracy, _, _ = _run_epoch(
        model, loader, loss_fn, optimizer
    )
    return loss, accuracy


def evaluate_position(model, loader, loss_fn):
    """Return loss, accuracy, reference labels, and predictions."""
    return _run_epoch(model, loader, loss_fn)


def run_seeded_cnn_epoch(model, loader, loss_fn, optimizer=None):
    """Train if an optimizer is supplied; otherwise evaluate."""
    loss, accuracy, _, _ = _run_epoch(
        model, loader, loss_fn, optimizer
    )
    return loss, accuracy


def run_seeded_position_epoch(model, loader, loss_fn, optimizer=None):
    """Train if an optimizer is supplied; otherwise evaluate."""
    loss, accuracy, _, _ = _run_epoch(
        model, loader, loss_fn, optimizer
    )
    return loss, accuracy


def train_weighted_epoch(
    model,
    loader,
    optimizer,
    weighted_loss_sum,
    class_weights,
):
    """Train using summed weighted cross-entropy.

    weighted_loss_sum must use reduction="sum" and the same
    class weights supplied through class_weights.
    """
    model.train()

    total_loss_sum = 0.0
    total_weight = 0.0
    total_correct = 0
    total_examples = 0

    for tokens, labels in loader:
        optimizer.zero_grad()

        scores = model(tokens)

        loss_sum = weighted_loss_sum(scores, labels)
        weight_sum = class_weights[labels].sum()
        loss = loss_sum / weight_sum

        loss.backward()
        optimizer.step()

        total_loss_sum += loss_sum.item()
        total_weight += weight_sum.item()
        total_correct += (
            scores.argmax(dim=1) == labels
        ).sum().item()
        total_examples += labels.size(0)

    if total_examples == 0 or total_weight <= 0:
        raise ValueError(
            "The loader must produce examples with positive total weight."
        )

    return (
        total_loss_sum / total_weight,
        total_correct / total_examples,
    )
