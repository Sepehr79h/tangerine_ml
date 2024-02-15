# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import torch

# # Load the tokenizer and model
# # tokenizer = AutoTokenizer.from_pretrained("./models/llama-2-chat-model")
# model = AutoModelForSequenceClassification.from_pretrained("./models/llama-2-chat-model")

# breakpoint()

# def inference(text, model, tokenizer):
#     # Preprocess input
#     prep_text = f'<startoftext>Content: {text}\nLabel: 0<endoftext>'
#     inputs = tokenizer(prep_text, return_tensors="pt")

#     # Perform inference
#     with torch.no_grad():
#         outputs = model(**inputs)

#     # Post-process outputs
#     logits = outputs.logits
#     predicted_class = torch.argmax(logits, dim=1).item()
#     return predicted_class

# input_text = "import pandas"
# predicted_class = inference(input_text, model, tokenizer)
# print("Predicted Class:", predicted_class)


# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# from fine_tune import load_dataset

# from transformers import Trainer, TrainingArguments
# import torch

# def test_model():
#     print("Load data")
#     tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
#     tokenizer.pad_token = tokenizer.eos_token
#     tokenizer.add_special_tokens({'pad_token': '<pad>'})
#     print("Tokenized data")
#     train_dataset, eval_dataset, test_dataset, force_ids = load_dataset(tokenizer)
#     # Load model again to see if it works
#     print("Loading model...")
#     saved_model = AutoModelForSequenceClassification.from_pretrained("./output/llama-2-chat-model")
#     print("Loading Successful")

#     # Initialize lists to store predictions and true labels
#     saved_preds = []
#     saved_labels = []

#     # Iterate over the test dataset
#     for i in range(len(test_dataset)):
#         # Get the input tensors and the true label
#         example = test_dataset[i]
#         inputs = {key: val.unsqueeze(0) for key, val in example.items() if key != 'labels'}
#         true_label = example['labels'].argmax().item()

#         # Perform inference
#         with torch.no_grad():
#             outputs = saved_model(**inputs)

#         # Get the predicted label
#         predicted_label = outputs.logits.argmax(-1).item()

#         # Store the predicted label and the true label
#         saved_preds.append(predicted_label)
#         saved_labels.append(true_label)

#     # Compute accuracy
#     correct_predictions = np.sum(np.array(saved_preds) == np.array(saved_labels))
#     total_predictions = len(saved_labels)
#     accuracy = correct_predictions / total_predictions
#     print("Test Accuracy: ", accuracy)

# test_model()



from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model

print("Loading Model...")
breakpoint()
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.add_special_tokens({'pad_token': '<pad>'})
# vocab_size = tokenizer.vocab_size
# model = AutoModelForSequenceClassification.from_pretrained("./output/llama-2-chat-model", num_labels=5, vocab_size=vocab_size,
#                                                            pad_token_id=tokenizer.eos_token_id)
# model.resize_token_embeddings(len(tokenizer))

# # LoRA Config
# peft_parameters = LoraConfig(
#     lora_alpha=16,
#     lora_dropout=0.1,
#     r=8,
#     bias="none",
#     task_type="CAUSAL_LM"
# )
# model = get_peft_model(model, peft_parameters)

# def inference(text):
#     # Preprocess input
#     prep_text = text
#     inputs = tokenizer(prep_text, return_tensors='pt')
#     # Perform inference
#     outputs = model(**inputs)
#     print(outputs.logits)
#     # Post-process outputs
#     predicted_class = outputs.logits.argmax(-1).item()
#     return predicted_class

# input_text = "import pandas"
# predicted_class = inference(input_text)
# print("Predicted Class:", predicted_class)

breakpoint()