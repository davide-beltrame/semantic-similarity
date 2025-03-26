#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import re
import argparse
from collections import Counter

# ===================== PARAMETERS =====================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "..", "data")
DUMP_DIR = os.path.join(CURRENT_DIR, "..", "dump")
DEV_OUTPUT_FILE = os.path.join(DUMP_DIR, "track1_dev.csv")
TEST_OUTPUT_FILE = os.path.join(DUMP_DIR, "track1_test.csv")
# ======================================================

def preprocess_text(text):
    """Simple preprocessing: lowercase, remove punctuation, normalize whitespace."""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
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
    """Load data for retrieval based on whether we're using dev or test."""
    # Create dump directory if it doesn't exist
    os.makedirs(DUMP_DIR, exist_ok=True)
    
    # Load training data
    print("Loading training data...")
    train_prompts = pd.read_csv(os.path.join(DATA_DIR, "train_prompts.csv"))
    train_responses = pd.read_csv(os.path.join(DATA_DIR, "train_responses.csv"))
    print(f"Loaded {len(train_prompts)} training prompts and {len(train_responses)} training responses")
    
    if use_dev:
        # For dev evaluation, use only train data as candidates
        candidate_prompts = train_prompts.copy()
        candidate_responses = train_responses.copy()
        
        # Load dev data as queries
        query_prompts = pd.read_csv(os.path.join(DATA_DIR, "dev_prompts.csv"))
        output_file = DEV_OUTPUT_FILE
        print(f"Dev evaluation mode: {len(candidate_prompts)} candidates (train only), {len(query_prompts)} queries (dev)")
    else:
        # For test submission, use train + dev as candidates
        dev_prompts = pd.read_csv(os.path.join(DATA_DIR, "dev_prompts.csv"))
        try:
            dev_responses = pd.read_csv(os.path.join(DATA_DIR, "dev_responses.csv"))
            print(f"Loaded {len(dev_prompts)} dev prompts and {len(dev_responses)} dev responses")
            
            # Combine train and dev data for candidates
            candidate_prompts = pd.concat([train_prompts, dev_prompts], ignore_index=True)
            candidate_responses = pd.concat([train_responses, dev_responses], ignore_index=True)
        except FileNotFoundError:
            print("Warning: Dev responses file not found, using only train data as candidates")
            candidate_prompts = pd.concat([train_prompts, dev_prompts], ignore_index=True)
            candidate_responses = train_responses.copy()
        
        # Load test data as queries
        query_prompts = pd.read_csv(os.path.join(DATA_DIR, "test_prompts.csv"))
        output_file = TEST_OUTPUT_FILE
        print(f"Test mode: {len(candidate_prompts)} candidates, {len(query_prompts)} queries (test)")
    
    # Performance optimization: create a dictionary mapping conversation_id to response
    response_dict = {}
    for _, row in candidate_responses.iterrows():
        response_dict[row['conversation_id']] = row['model_response']
    
    return candidate_prompts, response_dict, query_prompts, output_file

def build_tfidf_representation(prompts):
    """Build TF-IDF representation for prompts."""
    print("Building TF-IDF representation...")
    processed_prompts = [preprocess_text(prompt) for prompt in prompts]
    
    # Standard TF-IDF with good parameter values
    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),  # Include both unigrams and bigrams
        min_df=3,            # More aggressive filtering of rare terms
        max_df=0.8,          # More aggressive filtering of common terms
        max_features=5000    # Focus on most important features
    )
    
    prompt_vectors = vectorizer.fit_transform(processed_prompts)
    print(f"TF-IDF matrix shape: {prompt_vectors.shape}")
    
    return vectorizer, prompt_vectors

def find_best_match(query_prompt, vectorizer, candidate_vectors, candidate_prompts, response_dict):
    """Find most similar prompt with optimized BLEU potential."""
    # Get TF-IDF similarity
    processed_query_prompt = preprocess_text(query_prompt)
    query_vec = vectorizer.transform([processed_query_prompt])
    sims = cosine_similarity(query_vec, candidate_vectors).flatten()
    
    # Get initial candidates - more than we need for thorough screening
    top_n = 40  # Larger pool of candidates for better chance of finding good BLEU matches
    top_indices = sims.argsort()[-top_n:][::-1]
    
    # Extract content words from test prompt
    test_content_words = extract_content_words(query_prompt)
    
    # Score the candidates on multiple dimensions
    candidate_scores = []
    
    for idx in top_indices:
        train_prompt = candidate_prompts.iloc[idx]["user_prompt"]
        train_id = candidate_prompts.iloc[idx]["conversation_id"]
        
        # 1. TF-IDF similarity - base metric
        tfidf_score = sims[idx]
        
        # 2. Content word overlap - critical for semantic similarity
        train_content_words = extract_content_words(train_prompt)
        word_overlap = len(test_content_words.intersection(train_content_words)) / max(1, len(test_content_words))
        
        # 3. Response characteristics
        response_score = 0.5  # Default
        response = response_dict.get(train_id, "")
        
        if response:
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

def run_retrieval(use_dev=False):
    """Main retrieval function optimized for BLEU performance."""
    print(f"Running retrieval for {'dev' if use_dev else 'test'} dataset...")
    
    # 1) Load data
    candidate_prompts, response_dict, query_prompts, output_file = load_data(use_dev)
    
    # 2) Build TF-IDF representation
    vectorizer, candidate_vectors = build_tfidf_representation(candidate_prompts["user_prompt"])
    
    # 3) Find best matches for each query prompt
    results = []
    
    for i, row in query_prompts.iterrows():
        query_id = row["conversation_id"]
        query_prompt = row["user_prompt"]
        
        # Find best match optimized for BLEU potential
        best_idx = find_best_match(query_prompt, vectorizer, candidate_vectors, candidate_prompts, response_dict)
        best_id = candidate_prompts.iloc[best_idx]["conversation_id"]
        
        # Store result
        results.append({
            "conversation_id": query_id,
            "response_id": best_id
        })
        
        # Print progress
        if i % 1000 == 0 and i > 0:
            print(f"Processed {i} out of {len(query_prompts)} prompts")
    
    # 4) Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    
    return output_file

def main():
    parser = argparse.ArgumentParser(description="Track 1: Discrete Text Representation optimized for BLEU score")
    parser.add_argument("--use_dev", action="store_true", help="Use dev set for evaluation")
    args = parser.parse_args()
    
    run_retrieval(args.use_dev)

if __name__ == "__main__":
    main()