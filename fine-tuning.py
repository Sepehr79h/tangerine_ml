import os

from datasets import metric, load_metric
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, AdamW, TrainingArguments, Trainer
import torch.nn as nn
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from dataset import create_notebooks_data
from train import train, test
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW


class MyDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        return {
            'input_ids': np.array(text['input_ids']),
            'attention_mask': np.array(text['attention_mask']),
            'label': label
        }


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
notebooks_data = create_notebooks_data('notebooks.txt', 'id2stages.json')
code_text = [entry['context'] for entry in notebooks_data]
labels = [entry['stage'] for entry in notebooks_data]

# Replace with the appropriate GPT-2 model name you intend to fine-tune
model_name = "meta-llama/Llama-2-7b-chat-hf"  # Use the same model name as in your initial code
tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # You can choose a different token if needed
tokenized_code = [tokenizer(text, truncation=True, padding='max_length') for text in code_text]

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
# Split the data into training, validation, and test sets
train_texts, test_texts, train_labels, test_labels = train_test_split(tokenized_code, encoded_labels, test_size=0.3, random_state=42)
val_texts, test_texts, val_labels, test_labels = train_test_split(test_texts, test_labels, test_size=0.5, random_state=42)

train_dataset = MyDataset(train_texts, train_labels, tokenizer, max_length=128)
val_dataset = MyDataset(val_texts, val_labels, tokenizer, max_length=128)
test_dataset = MyDataset(test_texts, test_labels, tokenizer, max_length=128)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=64)

# Load a pre-trained GPT-2 model for fine-tuning
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)
model = model.to(device)

# optimizer = AdamW(model.parameters(), lr=1e-5)  # You can adjust the learning rate
# criterion = nn.CrossEntropyLoss()
# model, training_losses, validation_losses, validation_accuracies = train(model, train_loader, val_loader, optimizer, criterion, device, num_epochs=30)
# # After training, you can use the trained model for testing
# test_accuracy = test(model, test_loader, criterion, device)
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    correct_predictions = np.sum(predictions == labels)
    total_predictions = len(labels)
    accuracy = correct_predictions / total_predictions
    return accuracy

training_args = TrainingArguments(output_dir="test_trainer", overwrite_output_dir=True, logging_strategy="no",
                                  save_strategy="no", num_train_epochs=6, per_device_train_batch_size=1,
                                  per_device_eval_batch_size=1, evaluation_strategy="epoch")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

os.environ["WANDB_DISABLED"] = "true"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Start Training Model")
trainer.train()
print("Start Evaluating Model combined topics")
# predictions = trainer.predict(test_dataset)
# print(predictions.predictions.shape, predictions.label_ids.shape)
# labels = predictions.label_ids
# preds = np.argmax(predictions.predictions, axis=-1)
# print(metric.compute(predictions=preds, references=labels))
predictions = trainer.predict(test_dataset)
print(predictions.predictions.shape, predictions.label_ids.shape)
labels = predictions.label_ids
preds = np.argmax(predictions.predictions, axis=-1)

correct_predictions = np.sum(preds == labels)
total_predictions = len(labels)
accuracy = correct_predictions / total_predictions
print(accuracy)
