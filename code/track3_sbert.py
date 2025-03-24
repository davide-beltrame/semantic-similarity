#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# ===================== PARAMETERS =====================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "..", "data")
DUMP_DIR = os.path.join(CURRENT_DIR, "..", "dump")

# CSV file names
TRAIN_PROMPTS_FILE = os.path.join(DATA_DIR, "train_prompts.csv")
DEV_PROMPTS_FILE   = os.path.join(DATA_DIR, "dev_prompts.csv")
TEST_PROMPTS_FILE  = os.path.join(DATA_DIR, "test_prompts.csv")

# Model parameters
#MODEL_NAME = "all-MiniLM-L6-v2"  # 0.10233936335610462
MODEL_NAME = "all-mpnet-base-v2" # 0.10786308257756748
#MODEL_NAME = "paraphrase-mpnet-base-v2" # 0.10241946736959553
#MODEL_NAME = "all-distilroberta-v1" # 0.1028827366145876
#MODEL_NAME = "multi-qa-mpnet-base-dot-v1" # 0.10346886872926038
# ======================================================

def load_data(use_dev=False):
    """
    Loads candidate and query prompts.
    - For dev evaluation (use_dev=True): use only train as candidate pool, and dev as queries.
    - Otherwise (for test submission): candidate pool = train + dev, query = test prompts.
    """
    if use_dev:
        candidate_df = pd.read_csv(TRAIN_PROMPTS_FILE)
        query_df = pd.read_csv(DEV_PROMPTS_FILE)
    else:
        train_df = pd.read_csv(TRAIN_PROMPTS_FILE)
        dev_df = pd.read_csv(DEV_PROMPTS_FILE)
        candidate_df = pd.concat([train_df, dev_df], ignore_index=True)
        query_df = pd.read_csv(TEST_PROMPTS_FILE)
    return candidate_df, query_df

def encode_texts(model, texts):
    """
    Encodes a list/series of texts into embeddings.
    """
    return model.encode(texts.tolist(), show_progress_bar=True)

def retrieve_responses(candidate_df, query_df, use_dev=False):
    """
    For each query prompt, encode using Sentence-BERT and compute cosine similarity
    with candidate prompt embeddings. In dev mode, self-matches are excluded.
    Returns a DataFrame with query conversation_id and retrieved candidate's conversation_id.
    """
    print("Loading Sentence-BERT model...")
    model = SentenceTransformer(MODEL_NAME)
    
    print("Encoding candidate prompts...")
    candidate_texts = candidate_df["user_prompt"].fillna("").astype(str)
    candidate_embeddings = encode_texts(model, candidate_texts)
    
    print("Encoding query prompts...")
    query_texts = query_df["user_prompt"].fillna("").astype(str)
    query_embeddings = encode_texts(model, query_texts)
    
    print("Computing cosine similarities...")
    sims = cosine_similarity(query_embeddings, candidate_embeddings)
    
    if use_dev:
        # Exclude self-matches: for queries coming from dev, remove candidate with same conversation_id.
        candidate_ids = candidate_df["conversation_id"].tolist()
        query_ids = query_df["conversation_id"].tolist()
        for i, qid in enumerate(query_ids):
            for j, cid in enumerate(candidate_ids):
                if qid == cid:
                    sims[i, j] = -1  # force non-selection of self-match

    best_indices = np.argmax(sims, axis=1)
    retrieved_ids = candidate_df.iloc[best_indices]["conversation_id"].values
    
    output_df = pd.DataFrame({
        "conversation_id": query_df["conversation_id"],
        "response_id": retrieved_ids
    })
    return output_df

def main():
    parser = argparse.ArgumentParser(description="Track 3: Retrieval using Sentence-BERT embeddings")
    parser.add_argument("--use_dev", action="store_true",
                        help="Use the dev set as queries (for evaluation). Otherwise, use test set.")
    args = parser.parse_args()
    
    candidate_df, query_df = load_data(use_dev=args.use_dev)
    retrieval_df = retrieve_responses(candidate_df, query_df, use_dev=args.use_dev)
    
    if args.use_dev:
        output_file = os.path.join(DUMP_DIR, "track3_sbert_dev.csv")
    else:
        output_file = os.path.join(DUMP_DIR, "track3_sbert_test.csv")
    
    retrieval_df.to_csv(output_file, index=False)
    print(f"Saved retrieval results to {output_file}")

if __name__ == "__main__":
    main()
