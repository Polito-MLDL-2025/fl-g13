import torch


def eval(model, dataloader, criterion):
    """
    Evaluate the model on the given dataloader using the specified loss function.
    """
    device = next(model.parameters()).device
    model.eval()

    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = criterion(logits, y)

            total_loss += loss.item()
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.size(0)

    test_loss = total_loss / len(dataloader)
    test_accuracy = correct / total

    return test_loss, test_accuracy