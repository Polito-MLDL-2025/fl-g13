import torch

from fl_g13.modeling.load import save

def train_one_epoch(model, optimizer, dataloader, loss_fn, verbose=False):
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


def train_model(checkpoint_dir, dataloader, loss_fn, start_epoch, num_epochs, save_every, model, optimizer,
                scheduler=None, filename=None, verbose=False):
    for epoch in range(start_epoch, start_epoch + num_epochs):
        avg_loss, training_accuracy = train_one_epoch(model, optimizer, dataloader, loss_fn, verbose=verbose)
        print(f"📘 Epoch [{epoch}/{start_epoch + num_epochs - 1}] - Avg Loss: {avg_loss:.4f}, Accuracy: {training_accuracy:.2f}%")

        if scheduler:
            scheduler.step()

        if save_every and (epoch - start_epoch + 1) % save_every == 0:
            save(checkpoint_dir, model, optimizer, scheduler, epoch=epoch, filename=filename)
