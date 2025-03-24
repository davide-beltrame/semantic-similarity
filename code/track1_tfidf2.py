#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import Counter

# ===================== PARAMETERS =====================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "..", "data")
DUMP_DIR = os.path.join(CURRENT_DIR, "..", "dump")

# CSV file names
TRAIN_PROMPTS_FILE = os.path.join(DATA_DIR, "train_prompts.csv")
DEV_PROMPTS_FILE   = os.path.join(DATA_DIR, "dev_prompts.csv")
TEST_PROMPTS_FILE  = os.path.join(DATA_DIR, "test_prompts.csv")

# TF-IDF Parameters
MAX_FEATURES = 5000
NGRAM_RANGE = (1, 2)
# ======================================================

def preprocess_text(text):
    """Simple preprocessing: lowercase, remove punctuation, normalize whitespace."""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)  # Replace punctuation with spaces
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
    return text

def extract_content_words(text):
    """Extract important content words from text."""
    words = preprocess_text(text).split()
    # Filter out stopwords (simple approach)
    stopwords = {'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what', 
                'when', 'where', 'how', 'which', 'this', 'that', 'these', 'those', 
                'then', 'to', 'of', 'in', 'for', 'with', 'by', 'at', 'from', 'is', 
                'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does',
                'did', 'can', 'could', 'will', 'would', 'should', 'i', 'you', 'he', 
                'she', 'it', 'we', 'they', 'their', 'my', 'your', 'his', 'her', 'its',
                'our', 'their'}
    content_words = [word for word in words if word not in stopwords and len(word) > 2]
    return set(content_words)

def load_data(use_dev=False):
    """
    Loads candidate and query prompts.
    - For dev evaluation (use_dev=True): use only train as candidate pool, and dev as queries.
    - Otherwise (for test submission): candidate pool = train + dev, query = test prompts.
    """
    # Load training data
    train_df = pd.read_csv(TRAIN_PROMPTS_FILE)
    
    # Load train responses if available
    train_responses_path = os.path.join(DATA_DIR, "train_responses.csv")
    try:
        train_responses_df = pd.read_csv(train_responses_path)
        print(f"Loaded {len(train_responses_df)} training responses")
    except FileNotFoundError:
        print("Warning: Training responses file not found")
        train_responses_df = None

    if use_dev:
        candidate_df = train_df
        query_df = pd.read_csv(DEV_PROMPTS_FILE)
        responses_df = train_responses_df
    else:
        dev_df = pd.read_csv(DEV_PROMPTS_FILE)
        candidate_df = pd.concat([train_df, dev_df], ignore_index=True)
        query_df = pd.read_csv(TEST_PROMPTS_FILE)
        
        # Load dev responses if available
        dev_responses_path = os.path.join(DATA_DIR, "dev_responses.csv")
        try:
            dev_responses_df = pd.read_csv(dev_responses_path)
            if train_responses_df is not None:
                responses_df = pd.concat([train_responses_df, dev_responses_df], ignore_index=True)
            else:
                responses_df = dev_responses_df
            print(f"Loaded {len(dev_responses_df)} dev responses")
        except FileNotFoundError:
            print("Warning: Dev responses file not found")
            responses_df = train_responses_df
    
    return candidate_df, query_df, responses_df

def find_best_match(test_prompt, vectorizer, train_vectors, candidate_df, responses_df=None, use_dev=False):
    """Find most similar prompt with optimized BLEU potential."""
    # Get TF-IDF similarity
    processed_test_prompt = preprocess_text(test_prompt)
    test_vec = vectorizer.transform([processed_test_prompt])
    sims = cosine_similarity(test_vec, train_vectors).flatten()
    
    # For dev set, mask self-matches by setting similarity to -1
    if use_dev:
        candidate_ids = candidate_df["conversation_id"].tolist()
        for i, cid in enumerate(candidate_ids):
            if cid == test_id:  # test_id is from the outer scope
                sims[i] = -1
    
    # Get initial candidates - more than we need for thorough screening
    top_n = min(40, len(sims))  # In case we have fewer than 40 candidates
    top_indices = sims.argsort()[-top_n:][::-1]
    
    # Extract content words from test prompt
    test_content_words = extract_content_words(test_prompt)
    
    # Score the candidates on multiple dimensions
    candidate_scores = []
    
    for idx in top_indices:
        train_prompt = candidate_df.iloc[idx]["user_prompt"]
        train_id = candidate_df.iloc[idx]["conversation_id"]
        
        # 1. TF-IDF similarity - base metric
        tfidf_score = sims[idx]
        
        # 2. Content word overlap - critical for semantic similarity
        train_content_words = extract_content_words(train_prompt)
        word_overlap = len(test_content_words.intersection(train_content_words)) / max(1, len(test_content_words))
        
        # 3. Response characteristics (if responses are available)
        response_score = 0.5  # Default
        if responses_df is not None:
            response_rows = responses_df[responses_df["conversation_id"] == train_id]
            if not response_rows.empty:
                response = str(response_rows.iloc[0]["model_response"])
                
                # Calculate response quality factors correlated with good BLEU scores
                
                # 3a. Response length (moderate length responses tend to have better BLEU scores)
                resp_len = len(response.split())
                length_score = 0.0
                if resp_len > 20 and resp_len < 300:  # Sweet spot for response length
                    length_score = 0.8
                elif resp_len <= 20:  # Very short responses
                    length_score = 0.3
                else:  # Very long responses
                    length_score = 0.5
                
                # 3b. Response specificity (responses with content words from the prompt tend to be better)
                content_word_count = len(set(response.lower().split()).intersection(test_content_words))
                specificity_score = min(1.0, content_word_count / max(1, len(test_content_words)))
                
                # 3c. Response is not too generic (helps avoid "I don't know" type responses)
                generic_phrases = ['i dont know', 'cannot', 'sorry', 'ai', 'language model']
                generic_score = 1.0
                for phrase in generic_phrases:
                    if phrase in response.lower():
                        generic_score *= 0.8  # Penalize generic responses
                
                # Combine response factors
                response_score = (length_score + specificity_score + generic_score) / 3
        
        # Calculate final combined score for this candidate
        # Weight content word overlap more heavily as it correlates better with BLEU
        combined_score = (
            0.35 * tfidf_score +      # Base similarity
            0.45 * word_overlap +     # Content overlap (most important)
            0.20 * response_score     # Response quality
        )
        
        candidate_scores.append((idx, combined_score))
    
    # Sort by combined score and select the best
    candidate_scores.sort(key=lambda x: x[1], reverse=True)
    best_idx = candidate_scores[0][0]
    
    return best_idx

def retrieve_responses(candidate_df, query_df, responses_df=None, use_dev=False):
    """
    For each query prompt, vectorize using TF-IDF and compute optimized similarity
    with candidate prompt vectorizations. Returns a DataFrame with query conversation_id 
    and retrieved candidate's conversation_id.
    """
    print("Preprocessing texts...")
    candidate_texts = candidate_df["user_prompt"].fillna("").astype(str)
    candidate_texts = candidate_texts.apply(preprocess_text)
    
    query_texts = query_df["user_prompt"].fillna("").astype(str)
    query_texts = query_texts.apply(preprocess_text)
    
    print("Building TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        stop_words='english',
        min_df=3,            # More aggressive filtering of rare terms
        max_df=0.8,          # More aggressive filtering of common terms
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=True
    )
    
    # Fit and transform on candidate texts
    candidate_vectors = vectorizer.fit_transform(candidate_texts)
    print(f"TF-IDF matrix shape: {candidate_vectors.shape}")
    
    # Process each query prompt
    results = []
    global test_id  # Used for self-match exclusion in find_best_match
    
    for i, row in query_df.iterrows():
        test_id = row["conversation_id"]
        test_prompt = row["user_prompt"]
        
        # Find best match optimized for BLEU potential
        best_idx = find_best_match(
            test_prompt, 
            vectorizer, 
            candidate_vectors, 
            candidate_df, 
            responses_df, 
            use_dev
        )
        best_id = candidate_df.iloc[best_idx]["conversation_id"]
        
        # Store result
        results.append({
            "conversation_id": test_id,
            "response_id": best_id
        })
        
        # Print progress
        if i % 1000 == 0 and i > 0:
            print(f"Processed {i} out of {len(query_df)} prompts")
    
    output_df = pd.DataFrame(results)
    return output_df

def main():
    parser = argparse.ArgumentParser(description="Track 1: Retrieval using TF-IDF vectorization")
    parser.add_argument("--use_dev", action="store_true",
                        help="Use the dev set as queries (for evaluation). Otherwise, use test set.")
    args = parser.parse_args()
    
    # Load data
    candidate_df, query_df, responses_df = load_data(use_dev=args.use_dev)
    print(f"Loaded {len(candidate_df)} prompts in retrieval pool")
    print(f"Loaded {len(query_df)} query prompts")
    
    # Retrieve responses
    retrieval_df = retrieve_responses(
        candidate_df, 
        query_df, 
        responses_df, 
        use_dev=args.use_dev
    )
    
    # Extract script name without extension for the output filename
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    
    if args.use_dev:
        output_file = os.path.join(DUMP_DIR, f"{script_name}_dev.csv")
    else:
        output_file = os.path.join(DUMP_DIR, f"{script_name}_test.csv")
    
    # Ensure dump directory exists
    os.makedirs(DUMP_DIR, exist_ok=True)
    
    # Save results
    retrieval_df.to_csv(output_file, index=False)
    print(f"Saved retrieval results to {output_file}")
    
    # Print statistics for dev set
    if args.use_dev:
        matches = sum(retrieval_df["conversation_id"] == retrieval_df["response_id"])
        print(f"Self-matches: {matches} out of {len(retrieval_df)} ({matches/len(retrieval_df)*100:.2f}%)")

if __name__ == "__main__":
    main()