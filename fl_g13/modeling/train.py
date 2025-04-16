import torch

from fl_g13.modeling.load import load_or_create_model, save_model


def _train(model, optimizer, dataloader, loss_fn, device, verbose=False):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()

        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(pred.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()

        if verbose:
            batch_acc = 100 * (predicted == y).sum().item() / y.size(0)
            print(f"  ↳ Batch {batch + 1}/{len(dataloader)} | Loss: {loss.item():.4f} | Batch Acc: {batch_acc:.2f}%")

    training_loss = total_loss / len(dataloader)
    training_accuracy = 100 * correct / total
    print(f"Training Loss: {training_loss:.4f}, Training Accuracy: {training_accuracy:.2f}%")
    return training_loss, training_accuracy


def train_model(checkpoint_dir, dataloader, loss_fn, num_epochs=100, save_every=10, lr=1e-4, weight_decay=0.04,
                model=None, optimizer=None, device=None, verbose=False):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, optimizer, start_epoch = load_or_create_model(
        checkpoint_dir=checkpoint_dir,
        model=model,
        optimizer=optimizer,
        lr=lr,
        weight_decay=weight_decay
    )

    for epoch in range(start_epoch, num_epochs + 1):
        avg_loss, training_accuracy = _train(model, optimizer, dataloader, loss_fn, device, verbose=verbose)
        print(f"📘 Epoch [{epoch}/{num_epochs}] - Avg Loss: {avg_loss:.4f}, Accuracy: {training_accuracy:.2f}%")

        if save_every and epoch % save_every == 0:
            save_model(model, optimizer, checkpoint_dir, epoch, prefix_name="dino_xcit")
