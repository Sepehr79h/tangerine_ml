import json
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from langchain.embeddings import LlamaCppEmbeddings
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tqdm import tqdm


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
    return embeddings


def create_dataset(notebooks_data, llama_embeddings):
    # Split data into training, validation, and test sets
    train_data, test_data = train_test_split(notebooks_data, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=0.25,
                                            random_state=42)
    # Convert labels to integer values
    label_encoder = LabelEncoder()
    train_labels_encoded = label_encoder.fit_transform([item['stage'] for item in train_data])
    test_labels_encoded = label_encoder.transform([item['stage'] for item in test_data])
    val_labels_encoded = label_encoder.transform([item['stage'] for item in val_data])
    # One-hot encoding
    train_labels_onehot = to_categorical(train_labels_encoded)
    test_labels_onehot = to_categorical(test_labels_encoded)
    val_labels_onehot = to_categorical(val_labels_encoded)
    # Convert each code in train, test, and validation sets to their corresponding embeddings
    train_embeddings = [llama_embeddings.embed_query(item['context']) for item in tqdm(train_data)]
    test_embeddings = [llama_embeddings.embed_query(item['context']) for item in tqdm(test_data)]
    val_embeddings = [llama_embeddings.embed_query(item['context']) for item in tqdm(val_data)]
    return train_embeddings, train_labels_onehot, test_embeddings, test_labels_onehot, val_embeddings, val_labels_onehot



