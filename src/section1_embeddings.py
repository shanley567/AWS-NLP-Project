def word_embeddings():
    import os
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    import kagglehub
    import tensorflow_hub as hub
    import numpy as np

    # Set the dataset path from an environment variable or use a default path
    DATASET_PATH = os.getenv(
        "DATASET_PATH",
        "/data/dataset.csv"
    )
    print(f"[INFO] - Dataset located at {DATASET_PATH}")


    # Download the model using kagglehub for better performance on Kaggle
    model_path = kagglehub.model_download(
        "google/wiki-words/tensorFlow2/250")

    print(f"[INFO] - Model Path: {model_path}")

    # Load the 250-dimensional Wiki-words embedding model from tensorFlow hub
    embed = hub.load(model_path)
    print(f"[INFO] - Model loaded successfully")
    print(f"[INFO] - Type of embed: {type(embed)}")



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
    print(f"[INFO] - Shape of our simple sentence embeddings: {sentence_embeddings.shape}")


    #############################################
    # Now use PCA to shrink the dimensionality of our sentence embeddings to 2D for visualization
    ##############################################
    from sklearn.decomposition import PCA

    # Reduce the 250-dimensional vectors down to 2 dimensions
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(sentence_embeddings)

    print(f"[INFO] - Shape after PCA reduction: {embeddings_2d.shape}")


    #############################################
    # Viewing the plot of the 2D embeddings
    #############################################
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))

    # Plot each simple sentence
    for i, label in enumerate(simple_labels):
        color = 'blue' if label == 1 else 'red'
        plt.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1], color=color, s=150)
        plt.annotate(simple_sentences[i], (embeddings_2d[i, 0], embeddings_2d[i, 1]), 
                    textcoords="offset points", xytext=(0,15), ha='center', color=color, fontsize=12)

    plt.title("2D PCA Visualization of Simple Sentence Embeddings")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True, linestyle='--', alpha=0.7)
    os.makedirs("/output", exist_ok=True)
    plt.savefig("/output/plot.png")
    print(f"[INFO] - Graph saved to '/output/plot.png'")


    #############################################
    # Calculating Cosine Similarity Between Words
    #############################################
    def calc_cosine_similarity(A, B):
        """Calculates the cosine similarity between two vectors."""
        return np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))

    # 1. Get embeddings for our simple words
    # Calling .numpy() converts the TensorFlow tensor back to a standard NumPy array
    excellent_vector = embed(["excellent"])[0].numpy()
    great_vector = embed(["great"])[0].numpy()
    terrible_vector = embed(["terrible"])[0].numpy()

    # 2. Calculate the Cosine Similarities
    sim_excellent_great = calc_cosine_similarity(excellent_vector, great_vector)
    sim_excellent_terrible = calc_cosine_similarity(excellent_vector, terrible_vector)

    print(f"[INFO] - Similarity between 'excellent' and 'great': {sim_excellent_great:.4f}")
    print(f"[INFO] - Similarity between 'excellent' and 'terrible': {sim_excellent_terrible:.4f}")