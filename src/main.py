from section1_embeddings import word_embeddings
from section2_data_prep import data_prep, get_word2vec_enc, get_max_length

###########################################
## Section 1: Word Embeddings
###########################################
# def sec1():
#     print("[INFO] - Section 1: Word Embeddings...")
#     word_embeddings()


###########################################
## Section 2: Data Preperation
###########################################
def sec2():
    print(
        f"""===============================================\n
        [INFO] - Section 2: Data Preperation
        \n===============================================\n"""
        )
    sample_size=2000

    df = data_prep().head(sample_size)

    max_length = get_max_length(df)
    
    
    encoded_texts = get_word2vec_enc(
            df["text"]
        )

    print(
        f"[INFO] Generated embeddings for "
        f"{len(encoded_texts)} texts"
        )


if __name__ == "__main__":
    sec2()


