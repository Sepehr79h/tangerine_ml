import torch.optim as optim

def train(model, train_X, train_y, val_X, val_y):
    train_X_torch = torch.tensor(train_X, dtype=torch.float32)
    train_y_torch = torch.tensor(train_y, dtype=torch.float32)
    val_X_torch = torch.tensor(val_X, dtype=torch.float32)
    val_y_torch = torch.tensor(val_y, dtype=torch.float32)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    training_losses = []
    validation_losses = []
    validation_accuracies = []

    num_epochs = 30
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(train_X_torch)
        loss = criterion(outputs, torch.argmax(train_y_torch, dim=1))
        loss.backward()
        optimizer.step()
        training_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_outputs = model(val_X_torch)
            val_loss = criterion(val_outputs, torch.argmax(val_y_torch, dim=1))
            validation_losses.append(val_loss.item())
            val_preds = torch.argmax(val_outputs, dim=1)
            val_labels = torch.argmax(val_y_torch, dim=1)
            val_accuracy = (val_preds == val_labels).sum().float() / len(val_labels)
            validation_accuracies.append(val_accuracy.item())

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Val Accuracy: {val_accuracy:.4f}")

    return model, training_losses, validation_losses, validation_accuracies

def test(model, test_X, test_y):
    test_X_torch = torch.tensor(test_X, dtype=torch.float32)
    test_y_torch = torch.tensor(test_y, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        test_outputs = model(test_X_torch)
        test_preds = torch.argmax(test_outputs, dim=1)
        test_labels = torch.argmax(test_y_torch, dim=1)
        test_accuracy = (test_preds == test_labels).sum().float() / len(test_labels)

    print(f"Test accuracy: {test_accuracy * 100:.2f}%")
    return test_accuracy.item()
