import torch

from fl_g13.modeling.load import save
from fl_g13.modeling.test import test
from fl_g13.modeling.utils import generate_goofy_name


def train_one_epoch(model, optimizer, dataloader, criterion, verbose=False):
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
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(pred, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()

        if verbose:
            batch_acc = correct / total
            print(
                f"  ↳ Batch {batch + 1}/{len(dataloader)} | Loss: {loss.item():.4f} | Batch Acc: {100 * batch_acc:.2f}%"
            )

    training_loss = total_loss / len(dataloader)
    training_accuracy = correct / total
    return training_loss, training_accuracy


def train(
    checkpoint_dir,
    train_dataloader,
    val_dataloader,
    criterion,
    start_epoch,
    num_epochs,
    save_every,
    model,
    optimizer,
    scheduler=None,
    prefix=None,
    verbose=False,
):
    """
    Trains a model for a specified number of epochs, saving checkpoints periodically.
    """

    if not prefix:
        prefix = generate_goofy_name(checkpoint_dir)

    for epoch in range(1, num_epochs + 1):
        # Train on the current epoch
        train_avg_loss, training_accuracy = train_one_epoch(
            model, optimizer, train_dataloader, criterion, verbose=verbose
        )
        print(
            f"🚀 Epoch [{epoch}/{num_epochs}] Completed\n"
            f"   📊 Training Loss: {train_avg_loss:.4f}\n"
            f"   ✅ Training Accuracy: {100 * training_accuracy:.2f}%"
        )
        
        # Immediately evaluate out-of-distribution accuracy
        test_avg_loss, validation_accuracy = test(
            model, val_dataloader, criterion
        )
        print(
            f"🔍 Validation Results:\n"
            f"   📉 Test Loss: {test_avg_loss:.4f}\n"
            f"   🎯 Test Accuracy: {100 * validation_accuracy:.2f}%"
        )

        if scheduler:
            scheduler.step()

        if save_every and epoch % save_every == 0:
            saving_epoch = start_epoch + epoch - 1
            save(checkpoint_dir, model, optimizer, scheduler, epoch=saving_epoch, prefix=prefix)
