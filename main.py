# from langchain.embeddings import LlamaCppEmbeddings
# from llama_cpp import LLama
#
# #Use Llama model for embedding
# llama_model_path = "../llama.cpp/models/llama-2-7b/ggml-model-q4_0.gguf"
# # embeddings = LlamaCppEmbeddings(model_path=llama_model_path)
#
# #If you want to specify the context window size for embedding, e.g. 2048
# embeddings = LlamaCppEmbeddings(model_path=llama_model_path, n_ctx=64, verbose=False)
#
# #Get embedding representation
# test_string = "import pandas as pd\n\n"
# test_string_embedding = embeddings.embed_query(test_string)
# breakpoint()
# print(test_string_embedding)

# from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
# import torch
#
# # Replace with the appropriate model name and tokenizer
# model_name = "gpt2"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # Add a padding token
#
# model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5, pad_token_id=tokenizer.eos_token_id)  # Assuming 5 labels
#
# # Code cell to classify
# code_cell = "import pandas as pd\n"
#
# # Tokenize the code cell
# tokens = tokenizer.encode(code_cell, add_special_tokens=True, padding=True, truncation=True, max_length=128, return_tensors="pt")
#
# # Perform inference
# with torch.no_grad():
#     outputs = model(tokens)
#     predicted_label = torch.argmax(outputs.logits, dim=1).item()
#
# # Map the predicted label to the corresponding category (e.g., Import, Wrangle, etc.)
# categories = ["Import", "Wrangle", "Explore", "Model", "Evaluate"]
# predicted_category = categories[predicted_label]
#
# print("Predicted category:", predicted_category)





import torch
from dataset import create_notebooks_data, get_llama_embeddings, create_dataset, create_dataset_sample
from models.SimpleNN import SimpleNN
from train import train, test

# Check for GPU availability and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

notebooks_data = create_notebooks_data('notebooks.txt', 'id2stages.json')
breakpoint()
llama_embeddings = get_llama_embeddings()
train_X, train_y, test_X, test_y, val_X, val_y = create_dataset(notebooks_data, llama_embeddings)
input_dim = train_X.shape[1]
hidden_dim = 128
output_dim = train_y.shape[1]

model = SimpleNN(input_dim, hidden_dim, output_dim).to(device)  # Move the model to the GPU if available
print("Training model...")
model, training_losses, validation_losses, validation_accuracies = train(model, train_X, train_y, val_X, val_y)
test_accuracy = test(model, test_X, test_y)

