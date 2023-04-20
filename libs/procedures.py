

def train(train_loader, optimizer, scheduler, num_epochs, device):
    """
    Train the model
    """
    # TODO: Implement this function (Task 4)
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train_one_epoch(model, train_loader, optimizer, device, epoch)
        scheduler.step()
        evaluate(model, val_loader, device)
    return model

