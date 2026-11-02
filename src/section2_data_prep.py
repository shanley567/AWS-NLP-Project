import os
import pandas as pd
import kagglehub
import tensorflow_hub as hub
print("[INFO] - Loading embedding model...")
model_path = kagglehub.model_download(
    "google/wiki-words/tensorFlow2/250"
)
embed = hub.load(model_path)
print("[INFO] - Embedding model loaded.")



def data_prep():


    dataset_path = os.getenv("DATASET_PATH")

    print(f"[INFO] - Loading dataset: {dataset_path}")

    df = pd.read_csv(dataset_path) # type: ignore

    # Standardise column names to tutorial format
    df = df.rename(
        columns={
            "clean_text": "text",
            "category": "label"
        }
    )

    # Remove missing values
    df = df.dropna().reset_index(drop=True)

    # Convert labels to integers
    df["label"] = df["label"].astype(int)

    print(
        f"[INFO] - Dataset shape after cleaning: {df.shape}"
    )

    labels = [int(x) for x in sorted(df["label"].unique())]
    print(
        f"[INFO] - Labels detected: {labels}"
    )

    print("\n========== CLASS DISTRIBUTION ==========")
    print(df["label"].value_counts())

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    return df

def get_max_length(df):
    """
    Get the maximum token count from the dataset.
    This number is used as a fixed-length input to the RNN cell.
    As RNN need a fixed sequence lenght
    """

    max_length = df["text"].str.split().str.len().max()
    print(f"[INFO] - Maximum token length in dataset: {max_length}")

    return max_length

def get_word2vec_enc(texts):

    encoded_texts = []

    print('\n[INFO] - Generating embeddings for texts...')

    for index, text in enumerate(texts):

        tokens = str(text).split()

        encoded_texts.append(
            embed(tokens)
        )

        if index % 100 == 0:
            print(
                f"[INFO] - Processed {index} rows"
            )

    return encoded_texts