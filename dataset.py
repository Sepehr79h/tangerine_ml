import json
# from sklearn.model_selection import train_test_split
# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
import os

import numpy as np
from langchain.embeddings import LlamaCppEmbeddings
from sklearn.preprocessing import LabelEncoder
# from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
import torch.nn.functional as F
import os
import numpy as np
from sklearn.model_selection import train_test_split


def create_notebooks_data(notebooks_path, labels_path):
    # Load the notebooks.txt data
    with open(notebooks_path, 'r') as f:
        notebooks_data = [json.loads(line) for line in f]
    # Load the id2stages.json data
    with open(labels_path, 'r') as f:
        id2stages_data = json.load(f)
    # Map the stages
    for entry in notebooks_data:
        notebook_id = entry['file']
        line_no = entry['target_lineno']
        # Minus 1 because lists are 0-indexed and line numbers start from 1
        stage = id2stages_data[notebook_id][line_no - 1]
        entry['stage'] = stage
    return notebooks_data


def get_llama_embeddings(llama_embeddings_path="../llama.cpp/models/llama-2-7b/ggml-model-q4_0.gguf"):
    # Use Llama model for embedding
    llama_model_path = llama_embeddings_path
    # embeddings = LlamaCppEmbeddings(model_path=llama_model_path)
    # If you want to specify the context window size for embedding, e.g. 2048
    embeddings = LlamaCppEmbeddings(model_path=llama_model_path, n_ctx=2048, verbose=False)
    # embeddings = LlamaCppEmbeddings(model_path=llama_model_path,
    #                                 n_ctx=2048,
    #                                 verbose=True,
    #                                 use_mlock=True,
    #                                 n_gpu_layers=8,
    #                                 n_threads=4,
    #                                 n_batch=1000)
    return embeddings


def create_dataset(notebooks_data, llama_embeddings):
    train_embeddings_file = 'embeddings/train_embeddings.npy'
    test_embeddings_file = 'embeddings/test_embeddings.npy'
    val_embeddings_file = 'embeddings/val_embeddings.npy'

    # Split data into training, validation, and test sets
    train_data, test_data = train_test_split(notebooks_data, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=0.25, random_state=42)
    # Convert labels to integer values
    label_encoder = LabelEncoder()
    train_labels_encoded = label_encoder.fit_transform([item['stage'] for item in train_data])
    test_labels_encoded = label_encoder.transform([item['stage'] for item in test_data])
    val_labels_encoded = label_encoder.transform([item['stage'] for item in val_data])
    # One-hot encoding
    train_labels_onehot = F.one_hot(torch.tensor(train_labels_encoded)).numpy()
    test_labels_onehot = F.one_hot(torch.tensor(test_labels_encoded)).numpy()
    val_labels_onehot = F.one_hot(torch.tensor(val_labels_encoded)).numpy()

    # Check if embeddings files exist
    if os.path.exists(train_embeddings_file) and os.path.exists(test_embeddings_file) and os.path.exists(val_embeddings_file):
        # Load saved embeddings
        print("Loading saved embeddings...")
        train_embeddings = np.load(train_embeddings_file)
        test_embeddings = np.load(test_embeddings_file)
        val_embeddings = np.load(val_embeddings_file)
    else:
        # Convert each code in train, test, and validation sets to their corresponding embeddings
        print("Creating embeddings...")
        train_embeddings = [llama_embeddings.embed_query(item['context']) for item in tqdm(train_data)]
        test_embeddings = [llama_embeddings.embed_query(item['context']) for item in tqdm(test_data)]
        val_embeddings = [llama_embeddings.embed_query(item['context']) for item in tqdm(val_data)]
        # Save the embeddings to files
        np.save(train_embeddings_file, train_embeddings)
        np.save(test_embeddings_file, test_embeddings)
        np.save(val_embeddings_file, val_embeddings)
    return train_embeddings, train_labels_onehot, test_embeddings, test_labels_onehot, val_embeddings, val_labels_onehot


def create_dataset_sample(notebooks_data, llama_embeddings, D=4096, num_samples=1840):
    train_embeddings_file = 'embeddings/train_embeddings.npy'
    test_embeddings_file = 'embeddings/test_embeddings.npy'
    val_embeddings_file = 'embeddings/val_embeddings.npy'

    # Split data into training, validation, and test sets
    train_data, test_data = train_test_split(notebooks_data, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=0.25, random_state=42)
    # Convert labels to integer values
    label_encoder = LabelEncoder()
    train_labels_encoded = label_encoder.fit_transform([item['stage'] for item in train_data])
    test_labels_encoded = label_encoder.transform([item['stage'] for item in test_data])
    val_labels_encoded = label_encoder.transform([item['stage'] for item in val_data])
    # One-hot encoding
    train_labels_onehot = F.one_hot(torch.tensor(train_labels_encoded)).numpy()
    test_labels_onehot = F.one_hot(torch.tensor(test_labels_encoded)).numpy()
    val_labels_onehot = F.one_hot(torch.tensor(val_labels_encoded)).numpy()

    # Check if embeddings files exist
    if os.path.exists(train_embeddings_file) and os.path.exists(test_embeddings_file) and os.path.exists(
            val_embeddings_file):
        # Load saved embeddings
        print("Loading saved embeddings...")
        train_embeddings = np.load(train_embeddings_file)
        test_embeddings = np.load(test_embeddings_file)
        val_embeddings = np.load(val_embeddings_file)
    else:
        # Generate random sample data
        train_X = np.random.rand(num_samples, D)
        train_y = np.random.randint(6, size=(num_samples, 6))

        # Split the data into training, validation, and test sets
        train_X, test_X, train_y, test_y = train_test_split(train_X, train_y, test_size=0.2, random_state=42)
        train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, test_size=0.25, random_state=42)
        train_embeddings = train_X
        test_embeddings = test_X
        val_embeddings = val_X
        # Save the generated data to files for future use
        np.save(train_embeddings_file, train_X)
        np.save(test_embeddings_file, test_X)
        np.save(val_embeddings_file, val_X)

    return train_embeddings, train_labels_onehot, test_embeddings, test_labels_onehot, val_embeddings, val_labels_onehot



