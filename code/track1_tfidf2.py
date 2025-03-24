#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import re
import string

# ===================== PARAMETERS =====================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "..", "data")
DUMP_DIR = os.path.join(CURRENT_DIR, "..", "dump")

# CSV file names
TRAIN_PROMPTS_FILE = os.path.join(DATA_DIR, "train_prompts.csv")
DEV_PROMPTS_FILE   = os.path.join(DATA_DIR, "dev_prompts.csv")
TEST_PROMPTS_FILE  = os.path.join(DATA_DIR, "test_prompts.csv")

# TF-IDF Parameters
MAX_FEATURES = 10000
NGRAM_RANGE = (1, 3)
APPLY_SVD = True
SVD_COMPONENTS = 300
# ======================================================

def simple_preprocess(text):
    """
    Simple text preprocessing without NLTK dependencies
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

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

def retrieve_responses(candidate_df, query_df, use_dev=False):
    """
    For each query prompt, vectorize using TF-IDF and compute cosine similarity
    with candidate prompt vectorizations. In dev mode, self-matches are excluded.
    Returns a DataFrame with query conversation_id and retrieved candidate's conversation_id.
    """
    print("Preprocessing texts...")
    candidate_texts = candidate_df["user_prompt"].fillna("").astype(str)
    candidate_texts = candidate_texts.apply(simple_preprocess)
    
    query_texts = query_df["user_prompt"].fillna("").astype(str)
    query_texts = query_texts.apply(simple_preprocess)
    
    print("Vectorizing with TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        stop_words='english',
        min_df=2,
        max_df=0.9,
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=True
    )
    
    # Fit and transform on candidate texts
    candidate_vectors = vectorizer.fit_transform(candidate_texts)
    
    # Transform query texts with the same vectorizer
    query_vectors = vectorizer.transform(query_texts)
    
    # Apply dimensionality reduction with SVD if configured
    if APPLY_SVD:
        print(f"Applying SVD to reduce dimensions to {SVD_COMPONENTS}...")
        svd = TruncatedSVD(n_components=SVD_COMPONENTS, random_state=42)
        candidate_vectors = svd.fit_transform(candidate_vectors)
        query_vectors = svd.transform(query_vectors)
    
    print("Computing cosine similarities...")
    # For dense vectors (SVD-transformed) or sparse matrices (more efficient calculation)
    sims = cosine_similarity(query_vectors, candidate_vectors)
    
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
    parser = argparse.ArgumentParser(description="Track 1: Retrieval using TF-IDF vectorization")
    parser.add_argument("--use_dev", action="store_true",
                        help="Use the dev set as queries (for evaluation). Otherwise, use test set.")
    args = parser.parse_args()
    
    candidate_df, query_df = load_data(use_dev=args.use_dev)
    retrieval_df = retrieve_responses(candidate_df, query_df, use_dev=args.use_dev)
    
    # Extract script name without extension for the output filename
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    
    if args.use_dev:
        output_file = os.path.join(DUMP_DIR, f"{script_name}_dev.csv")
    else:
        output_file = os.path.join(DUMP_DIR, f"{script_name}_test.csv")
    
    # Ensure dump directory exists
    os.makedirs(DUMP_DIR, exist_ok=True)
    
    retrieval_df.to_csv(output_file, index=False)
    print(f"Saved retrieval results to {output_file}")

if __name__ == "__main__":
    main()