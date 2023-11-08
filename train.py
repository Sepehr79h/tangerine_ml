import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader


def train(model, train_loader, val_loader, optimizer, criterion, device, num_epochs=30):
    model.train()
    model.to(device)

    training_losses = []
    validation_losses = []
    validation_accuracies = []

    for epoch in range(num_epochs):
        # Training loop
        training_loss = 0.0
        for batch in train_loader:
            #breakpoint()
            inputs = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            optimizer.zero_grad()
            breakpoint()
            outputs = model(input_ids=inputs, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()

            training_loss += loss.item()

        training_loss /= len(train_loader)
        training_losses.append(training_loss)

        # Validation loop
        model.eval()
        validation_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                breakpoint()
                outputs = model(input_ids=inputs, attention_mask=attention_mask, labels=labels)
                loss = criterion(outputs.logits, labels)
                validation_loss += loss.item()

                _, predicted = torch.max(outputs.logits, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_samples += labels.size(0)

        validation_loss /= len(val_loader)
        validation_losses.append(validation_loss)
        validation_accuracy = correct_predictions / total_samples
        validation_accuracies.append(validation_accuracy)

        print(
            f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {training_loss:.4f}, Validation Loss: {validation_loss:.4f}, Validation Accuracy: {validation_accuracy:.4f}")

    return model, training_losses, validation_losses, validation_accuracies


def test(model, test_loader, criterion, device):
    model.eval()
    model.to(device)

    test_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=inputs, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.logits, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    test_loss /= len(test_loader)
    test_accuracy = correct_predictions / total_samples

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    return test_accuracy
