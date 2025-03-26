#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ===================== PARAMETERS =====================
# Define base directory relative to this file
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "..", "data")
DUMP_DIR = os.path.join(CURRENT_DIR, "..", "dump")

# CSV file names
TRAIN_PROMPTS_FILE = os.path.join(DATA_DIR, "train_prompts.csv")
DEV_PROMPTS_FILE   = os.path.join(DATA_DIR, "dev_prompts.csv")
TEST_PROMPTS_FILE  = os.path.join(DATA_DIR, "test_prompts.csv")

# Vectorizer parameters - tuned for better semantic matching
NGRAM_RANGE = (1, 2)  # Use unigrams and bigrams
MAX_FEATURES = 10000  # Maximum number of features
# ======================================================

def preprocess_text(text):
    """Lightweight preprocessing for better matching."""
    if text is None or pd.isna(text):
        return ""
    text = str(text).lower()
    # Simple punctuation removal and whitespace normalization
    text = re.sub(r'[^\w\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def load_data(use_dev=False, use_test=True):
    """
    Loads data based on evaluation mode:
    - For dev evaluation (use_dev=True): use train as candidates, dev as queries
    - For test submission (use_dev=False, use_test=True): use train+dev as candidates, test as queries
    """
    print("Loading data...")
    train_prompts = pd.read_csv(TRAIN_PROMPTS_FILE)
    
    if use_dev:
        # For dev evaluation, use ONLY train data as candidates
        # This is crucial - we should NEVER use dev data as candidates when evaluating on dev
        candidate_prompts = train_prompts.copy()
        query_prompts = pd.read_csv(DEV_PROMPTS_FILE)
        print(f"Dev evaluation mode: {len(candidate_prompts)} candidates (train only), {len(query_prompts)} queries (dev)")
    else:
        # For test submission, use both train and dev as candidates
        dev_prompts = pd.read_csv(DEV_PROMPTS_FILE)
        candidate_prompts = pd.concat([train_prompts, dev_df], ignore_index=True)
        
        if use_test:
            query_prompts = pd.read_csv(TEST_PROMPTS_FILE)
            print(f"Test mode: {len(candidate_prompts)} candidates (train+dev), {len(query_prompts)} queries (test)")
        else:
            # For train evaluation (debugging)
            query_prompts = train_prompts.copy()
            print(f"Train mode: {len(candidate_prompts)} candidates (train+dev), {len(query_prompts)} queries (train)")
    
    return candidate_prompts, query_prompts

def retrieve_responses(candidate_prompts, query_prompts):
    """
    For each query prompt, finds the most similar candidate using TF-IDF and cosine similarity.
    Returns a DataFrame with the query conversation_id and the selected candidate's conversation_id.
    """
    print("Preprocessing texts...")
    # Preprocess all texts
    candidate_texts = [preprocess_text(text) for text in candidate_prompts['user_prompt']]
    query_texts = [preprocess_text(text) for text in query_prompts['user_prompt']]
    
    print("Building TF-IDF representation...")
    # Create and fit TF-IDF vectorizer on candidate prompts
    vectorizer = TfidfVectorizer(
        ngram_range=NGRAM_RANGE, 
        max_features=MAX_FEATURES,
        min_df=2,  # Ignore terms that appear in fewer than 2 documents
        max_df=0.9,  # Ignore terms that appear in more than 90% of documents
        stop_words='english'  # Remove common English stopwords
    )
    candidate_vectors = vectorizer.fit_transform(candidate_texts)
    
    # Transform query prompts into the same vector space
    query_vectors = vectorizer.transform(query_texts)
    
    print("Computing similarities...")
    # Process queries in batches to reduce memory usage
    batch_size = 1000
    num_queries = len(query_prompts)
    retrieved_ids = []
    
    for i in range(0, num_queries, batch_size):
        end_idx = min(i + batch_size, num_queries)
        print(f"Processing queries {i+1}-{end_idx} of {num_queries}...")
        
        # Compute cosine similarity for this batch
        batch_similarities = cosine_similarity(
            query_vectors[i:end_idx], 
            candidate_vectors
        )
        
        # For each query in the batch, get index of the candidate with highest similarity
        batch_best_indices = np.argmax(batch_similarities, axis=1)
        batch_retrieved_ids = candidate_prompts.iloc[batch_best_indices]['conversation_id'].values
        retrieved_ids.extend(batch_retrieved_ids)
    
    # Build output DataFrame
    output_df = pd.DataFrame({
        "conversation_id": query_prompts["conversation_id"],
        "response_id": retrieved_ids
    })
    
    # Count self-matches for debugging
    if set(query_prompts["conversation_id"]).intersection(set(candidate_prompts["conversation_id"])):
        self_matches = (output_df["conversation_id"] == output_df["response_id"]).sum()
        self_match_pct = (self_matches / len(output_df)) * 100
        print(f"Self-matches: {self_matches}/{len(output_df)} ({self_match_pct:.2f}%)")
        
        # This should be 0 for dev evaluation if we properly use only train candidates
        if self_matches > 0:
            print("Warning: Self-matches found. Check that candidate pool doesn't include test/dev data!")
    
    return output_df

def main():
    parser = argparse.ArgumentParser(description="Semantic similarity retrieval using TF-IDF")
    parser.add_argument("--use_dev", action="store_true",
                        help="Use the dev set as query (for evaluation). Otherwise, use test set.")
    parser.add_argument("--use_train", action="store_true",
                        help="Use the train set as query (for debugging).")
    args = parser.parse_args()
    
    # Make sure dump directory exists
    os.makedirs(DUMP_DIR, exist_ok=True)
    
    # Determine which datasets to use
    use_test = not (args.use_dev or args.use_train)
    candidate_prompts, query_prompts = load_data(use_dev=args.use_dev, use_test=use_test)
    
    # Perform retrieval
    retrieval_df = retrieve_responses(candidate_prompts, query_prompts)
    
    # Determine output file name based on mode
    if args.use_dev:
        output_file = os.path.join(DUMP_DIR, "track1_countvec_dev.csv")
    elif args.use_train:
        output_file = os.path.join(DUMP_DIR, "track1_countvec_train.csv")
    else:
        output_file = os.path.join(DUMP_DIR, "track1_countvec_test.csv")
    
    retrieval_df.to_csv(output_file, index=False)
    print(f"Saved retrieval results to {output_file}")

if __name__ == "__main__":
    main()