import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import transformers
from datasets import load_dataset, load_metric
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import numpy as np
from transformers import TrainingArguments, Trainer
from torch.utils.data import Dataset
import os
import torch
from dataset import create_notebooks_data


class SODataset(Dataset):
    def __init__(self, txt_list, label_list, tokenizer):
        self.input_ids = []
        self.attention_mask = []
        self.labels = []

        for txt, label in zip(txt_list, label_list):
            # breakpoint()
            prep_txt = f'<startoftext>Content: {txt}\nLabel: {label}<endoftext>'

            encodings_dict = tokenizer(prep_txt, truncation=True, padding="max_length")

            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attention_mask.append(torch.tensor(encodings_dict['attention_mask']))
            self.labels.append(label)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        dic = {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }
        # return self.input_ids[idx], self.attention_mask[idx], self.labels[idx]
        return dic


def load_dataset(tokenizer, notebooks_path='notebooks.txt', labels_path='id2stages.json'):
    # # Concatenate the Flask and Python datasets
    # train_path_flask = "../data/Documentations/flask/flask_train_category.csv"
    # train_path_python = "../data/Documentations/python/python_train_category.csv"
    # test_path_flask = "../data/StackOverflow/Flask/test_flask_category.csv"
    # test_path_python = "../data/StackOverflow/Python/test_python_category.csv"
    # df_train_flask = pd.read_csv(train_path_flask, sep=",", header=0, engine="python")
    # df_train_python = pd.read_csv(train_path_python, sep=",", header=0, engine="python")
    # df_test_flask = pd.read_csv(test_path_flask, sep=",", header=0, engine="python", encoding="latin-1")
    # df_test_python = pd.read_csv(test_path_python, sep=",", header=0, engine="python", encoding="latin-1")
    # df_train = pd.concat([df_train_flask, df_train_python])
    # df_test = pd.concat([df_test_flask, df_test_python])
    # # remove everything inside the code tag
    # df_test['content'] = df_test['content'].str.replace(r'<code>(.|\n)*?</code>', '')
    # # remove all the html tags from content column of df_test
    # df_test['content'] = df_test['content'].str.replace(r'<[^>]+>', '')
    # df_test['content'] = df_test['content'].str.replace(r'\n', '')
    # # save all the unique labels in a list
    # labels = df_train['label'].unique().tolist()
    # # split test data to eval and test with 10% and 90% respectively
    # df_eval = df_test.sample(frac=0.1, random_state=42)
    # df_test = df_test.drop(df_eval.index)
    notebooks_data = create_notebooks_data(notebooks_path, labels_path)
    code_text = [entry['context'] for entry in notebooks_data]
    labels = [entry['stage'] for entry in notebooks_data]
    train_texts, test_texts, train_labels, test_labels = train_test_split(code_text, labels, test_size=0.3,
                                                                          random_state=42)
    val_texts, test_texts, val_labels, test_labels = train_test_split(test_texts, test_labels, test_size=0.5,
                                                                      random_state=42)

    train_dataset = SODataset(train_texts, train_labels, tokenizer)
    eval_dataset = SODataset(val_texts, val_labels, tokenizer)
    test_dataset = SODataset(test_texts, test_labels, tokenizer)
    force_ids = tokenizer(labels, add_special_tokens=False).input_ids

    return train_dataset, eval_dataset, test_dataset, force_ids


if __name__ == '__main__':
    print("Load data")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    tokenizer.pad_token = tokenizer.eos_token
    print("Tokenized data")
    train_dataset, eval_dataset, test_dataset, force_ids = load_dataset(tokenizer)
    vocab_size = tokenizer.vocab_size
    print("Create Model")
    # change the model name and num_labels if needed
    model = AutoModelForSequenceClassification.from_pretrained("meta-llama/Llama-2-7b-chat-hf", num_labels=63, vocab_size=vocab_size,
                                                               pad_token_id=tokenizer.eos_token_id)
    metric = load_metric("accuracy")


    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)


    training_args = TrainingArguments(output_dir="test_trainer", overwrite_output_dir=True, logging_strategy="no",
                                      save_strategy="no", num_train_epochs=6, per_device_train_batch_size=1,
                                      per_device_eval_batch_size=1, evaluation_strategy="epoch")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    os.environ["WANDB_DISABLED"] = "true"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Start Training Model")
    trainer.train()
    print("Start Evaluating Model combined topics")
    predictions = trainer.predict(test_dataset)
    print(predictions.predictions.shape, predictions.label_ids.shape)

    labels = predictions.label_ids
    preds = np.argmax(predictions.predictions, axis=-1)

    print(metric.compute(predictions=preds, references=labels))

    # Save Prediction result
    print("Saving prediction results flask topic")
    df = pd.DataFrame({'label': labels, 'prediction': preds})
    df.to_csv('combined_topic_predictions.csv', index=False)

    print("Saving model...")

    # trainer.save_model("./gpt2model_str_label_topic")
    trainer.save_model("./models/combined_gpt2model_cd_topic")

    # Load model again to see if it works

    # print("Loading model...")
    # saved_model = AutoModelForSequenceClassification.from_pretrained("./gpt2model")
    # print("Loading Successful")
    # saved_predictions = saved_model.predict(test_dataset)
    # print(saved_predictions.predictions.shape, saved_predictions.label_ids.shape)
    # saved_labels = saved_predictions.label_ids
    # saved_preds = np.argmax(saved_predictions.predictions, axis=-1)
    # print(metric.compute(predictions=saved_preds, references=saved_labels))