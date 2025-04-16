import torch

from fl_g13.modeling.load import save

def train_one_epoch(model, optimizer, dataloader, loss_fn, verbose=False):
    """
    Trains the model for one epoch using the provided dataloader, optimizer, and loss function.
    """
    device = next(model.parameters()).device
    model.train()

    total_loss, correct, total = 0.0, 0, 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()

        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(pred, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()

        if verbose:
            batch_acc = correct / total
            print(f"  ↳ Batch {batch + 1}/{len(dataloader)} | Loss: {loss.item():.4f} | Batch Acc: {100*batch_acc:.2f}%")

    training_loss = total_loss / len(dataloader)
    training_accuracy = correct / total
    print(f"Training Loss: {training_loss:.4f}, Training Accuracy: {100*training_accuracy:.2f}%")
    return training_loss, training_accuracy


def train(checkpoint_dir, dataloader, loss_fn, start_epoch, num_epochs, save_every, model, optimizer,
            scheduler=None, filename=None, verbose=False):
    """
    Trains a model for a specified number of epochs, saving checkpoints periodically.
    """
    for epoch in range(1, num_epochs+1):
        avg_loss, training_accuracy = train_one_epoch(model, optimizer, dataloader, loss_fn, verbose=verbose)
        print(f"📘 Epoch [{epoch}/{num_epochs}] - Avg Loss: {avg_loss:.4f}, Accuracy: {100*training_accuracy:.2f}%")

        if scheduler:
            scheduler.step()

        if save_every and epoch % save_every == 0:
            save(checkpoint_dir, model, optimizer, scheduler, epoch=(start_epoch+epoch), filename=filename)
