import torch


def test(model, dataloader, loss_fn):
    """
    Evaluate the model on the given dataloader using the specified loss function.
    """
    device = next(model.parameters()).device
    print(f"Using device: {device}")

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0
    preds, labels, probs, inputs = [], [], [], []

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)

            total_loss += loss.item()
            total += y.size(0)
            correct += (pred.argmax(dim=1) == y).sum().item()

            inputs.extend(X.cpu())
            preds.extend(pred.argmax(dim=1).cpu())
            labels.extend(y.cpu())
            probs.extend(pred.max(dim=1).values.cpu())

    test_loss = total_loss / len(dataloader)
    test_accuracy = 100 * correct / total
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

    return (
        torch.tensor(preds),
        torch.tensor(labels),
        torch.tensor(probs),
        inputs,
        test_accuracy,
    )