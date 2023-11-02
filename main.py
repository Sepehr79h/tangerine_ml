# from langchain.embeddings import LlamaCppEmbeddings
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
import torch
from dataset import create_notebooks_data, get_llama_embeddings, create_dataset
from models.SimpleNN import SimpleNN
from train import train, test

# Check for GPU availability and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

notebooks_data = create_notebooks_data('notebooks.txt', 'id2stages.json')
llama_embeddings = get_llama_embeddings()
train_X, train_y, test_X, test_y, val_X, val_y = create_dataset(notebooks_data, llama_embeddings)
input_dim = train_X.shape[1]
hidden_dim = 128
output_dim = train_y.shape[1]

model = SimpleNN(input_dim, hidden_dim, output_dim).to(device)  # Move the model to the GPU if available

model, training_losses, validation_losses, validation_accuracies = train(model, train_X, train_y, val_X, val_y, device)
test_accuracy = test(model, test_X, test_y, device)

