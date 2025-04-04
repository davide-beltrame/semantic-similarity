#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
import re
from gensim.models import FastText
from sklearn.metrics.pairwise import cosine_similarity

# ===================== PARAMETERS =====================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "..", "data")
DUMP_DIR = os.path.join(CURRENT_DIR, "..", "dump")

TRAIN_PROMPTS_FILE = os.path.join(DATA_DIR, "train_prompts.csv")
DEV_PROMPTS_FILE   = os.path.join(DATA_DIR, "dev_prompts.csv")
TEST_PROMPTS_FILE  = os.path.join(DATA_DIR, "test_prompts.csv")
# ======================================================

def preprocess_text(text):
    """Lowercase text, remove punctuation, and normalize whitespace."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def tokenize(text):
    """Simple whitespace tokenizer."""
    return text.split()

def load_data(use_dev=False):
    """
    Loads candidate and query prompts.
    For dev: candidate pool = train prompts, query = dev prompts.
    For test: candidate pool = train + dev prompts, query = test prompts.
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

def train_fasttext_model(texts):
    """
    Train a FastText model on tokenized texts.
    """
    tokenized_texts = [tokenize(preprocess_text(text)) for text in texts]
    # Parameters: vector_size=100, window=5, min_count=1 for a fast model.
    model = FastText(sentences=tokenized_texts, vector_size=100, window=5, min_count=1, workers=4)
    return model

def compute_average_vector(model, text):
    """
    Compute the average word vector for a given text using the FastText model.
    """
    tokens = tokenize(preprocess_text(text))
    vectors = [model.wv[token] for token in tokens if token in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

def build_representations(model, df):
    """
    Build a 2D numpy array where each row is the average FastText vector for a prompt.
    """
    vectors = df["user_prompt"].fillna("").apply(lambda x: compute_average_vector(model, x))
    return np.stack(vectors.to_numpy())

def retrieve_responses(candidate_df, query_df, use_dev=False):
    """
    For each query prompt, compute its continuous representation using FastText,
    compare it to candidate representations using cosine similarity, and return the
    candidate with the highest similarity. In dev mode, self-matches are excluded.
    """
    print("Training FastText model on candidate pool...")
    model = train_fasttext_model(candidate_df["user_prompt"].fillna("").tolist())
    
    print("Building candidate representations...")
    candidate_vectors = build_representations(model, candidate_df)
    
    print("Building query representations...")
    query_vectors = build_representations(model, query_df)
    
    print("Computing cosine similarities...")
    sims = cosine_similarity(query_vectors, candidate_vectors)
    
    if use_dev:
        # For dev queries, exclude self-match by setting similarity to -1.
        candidate_ids = candidate_df["conversation_id"].tolist()
        query_ids = query_df["conversation_id"].tolist()
        for i, qid in enumerate(query_ids):
            for j, cid in enumerate(candidate_ids):
                if qid == cid:
                    sims[i, j] = -1
    
    best_indices = np.argmax(sims, axis=1)
    retrieved_ids = candidate_df.iloc[best_indices]["conversation_id"].values
    output_df = pd.DataFrame({
        "conversation_id": query_df["conversation_id"],
        "response_id": retrieved_ids
    })
    return output_df

def main():
    parser = argparse.ArgumentParser(description="Track 2: Retrieval using FastText embeddings")
    parser.add_argument("--use_dev", action="store_true",
                        help="Use the dev set as queries (for evaluation). Otherwise, use test set.")
    args = parser.parse_args()
    
    candidate_df, query_df = load_data(use_dev=args.use_dev)
    retrieval_df = retrieve_responses(candidate_df, query_df, use_dev=args.use_dev)
    
    if args.use_dev:
        output_file = os.path.join(DUMP_DIR, "track2_fasttext_dev.csv")
    else:
        output_file = os.path.join(DUMP_DIR, "track2_fasttext_test.csv")
    
    retrieval_df.to_csv(output_file, index=False)
    print(f"Saved retrieval results to {output_file}")

if __name__ == "__main__":
    main()
