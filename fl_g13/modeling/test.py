import torch


def test(model, dataloader, loss_fn,device=None):
    # TODO
    model.eval()
    if not device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    total_loss = 0.0
    correct = 0
    total = 0
    preds = []
    labels = []
    probs = []
    inputs = []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)


            inputs.extend(X.cpu())
            total_loss += loss.item()
            original_pre = pred.cpu()
            prob, predicted = torch.max(pred.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            preds.extend(predicted.cpu())
            labels.extend(y.cpu())
            probs.extend(prob.cpu())

    test_loss = total_loss / len(dataloader)
    test_accuracy = 100 * correct / total
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
    preds = torch.tensor(preds)
    labels = torch.tensor(labels)
    probs = torch.tensor(probs)
    return preds, labels, probs, inputs, test_accuracy