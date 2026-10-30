import kagglehub
import tensorflow_hub as hub
import numpy as np
import os

# Set the dataset path from an environment variable or use a default path
DATASET_PATH = os.getenv(
    "DATASET_PATH",
    "/data/dataset.csv"
)
print(f"Dataset located at {DATASET_PATH}")


# Download the model using kagglehub for better performance on Kaggle
model_path = kagglehub.model_download(
    "google/wiki-words/tensorFlow2/250")

# Load the 250-dimensional Wiki-words embedding model from tensorFlow hub
embed = hub.load(model_path)
print("Model loaded successfully")
print(type(embed))



# 1. Define simple positive and negative sentences
simple_sentences = [
    "excellent",
    "great",
    "terrible",
    "awful"
]
# 1 for positive, 0 for negative
simple_labels = [1, 1, 0, 0] 

# 2. Get embeddings using our previously defined function
def get_word2vec_enc(texts):
    """
    Get the 250-dimensional vector value for each word in the sentence.
    """
    encoded_texts = []
    for text in texts:
        tokens = text.split(" ")
        word2vec_embedding = embed(tokens)
        encoded_texts.append(word2vec_embedding)
    return encoded_texts

simple_encoded = get_word2vec_enc(simple_sentences)

# 3. Calculate a single "sentence vector" for each by taking the mean of its words
sentence_embeddings = np.array([np.mean(enc, axis=0) for enc in simple_encoded])
print(f"Shape of our simple sentence embeddings: {sentence_embeddings.shape}")