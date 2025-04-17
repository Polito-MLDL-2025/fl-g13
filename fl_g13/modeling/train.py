import torch

from fl_g13.modeling.load import save
from fl_g13.modeling.eval import eval
from fl_g13.modeling.utils import generate_goofy_name


def train_one_epoch(
    dataloader, 
    model, 
    criterion, 
    optimizer, 
    verbose=False
):
    """
    Trains the model for one epoch using the provided dataloader, optimizer, and loss function.
    """
    device = next(model.parameters()).device
    model.train()

    total_loss, correct, total = 0.0, 0, 0
    for batch_idx, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()

        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

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
    training_avg_loss = total_loss / len(dataloader)
    training_accuracy = correct / total
    return training_avg_loss, training_accuracy


def train(
    checkpoint_dir,
    prefix,
    start_epoch,
    num_epochs,
    save_every,
    train_dataloader,
    val_dataloader,
    model,
    criterion,
    optimizer,
    scheduler=None,
    verbose=False,
):
    """
    Trains a model for a specified number of epochs, saving checkpoints periodically.
    """

    if not prefix:
        prefix = generate_goofy_name(checkpoint_dir)
        print(f"No prefix/name for the model was provided, choosen prefix/name: {prefix}")

    for epoch in range(1, num_epochs + 1):
        # Train on the current epoch
        train_avg_loss, training_accuracy = train_one_epoch(
            dataloader=train_dataloader, model=model, criterion=criterion, optimizer=optimizer, verbose=verbose
        )
        print(
            f"🚀 Epoch [{epoch}/{num_epochs}] Completed ({100*epoch/num_epochs:.2f})\n"
            f"\t📊 Training Loss: {train_avg_loss:.4f}\n"
            f"\t✅ Training Accuracy: {100 * training_accuracy:.2f}%"
        )
        
        # Immediately evaluate out-of-distribution accuracy
        validation_avg_loss, validation_accuracy = eval(
            dataloader=val_dataloader, model=model, criterion=criterion
        )
        print(
            f"🔍 Validation Results:\n"
            f"\t📉 Validation Loss: {validation_avg_loss:.4f}\n"
            f"\t🎯 Validation Accuracy: {100 * validation_accuracy:.2f}%"
        )

        if scheduler:
            scheduler.step()

        if save_every and epoch % save_every == 0:
            saving_epoch = start_epoch + epoch - 1
            save(checkpoint_dir, prefix, model, optimizer, scheduler, epoch=saving_epoch)
