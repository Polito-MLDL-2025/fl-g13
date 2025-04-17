import torch


def eval(dataloader, model, criterion, verbose=False):
    """
    Evaluate the model on the given dataloader using the specified loss function.
    If verbose is True, print progress and intermediate results.
    """
    device = next(model.parameters()).device
    model.eval()

    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = criterion(logits, y)

            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            batch_correct = (predicted == y).sum().item()
            batch_total = y.size(0)

            correct += batch_correct
            total += batch_total

            if verbose and batch_idx % 10 == 0:
                print(
                    f"  ↳ Batch {batch_idx + 1}/{len(dataloader)} | Loss: {loss.item():.4f}"
                )

    test_loss = total_loss / len(dataloader)
    test_accuracy = correct / total

    return test_loss, test_accuracy