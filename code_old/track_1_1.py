#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from collections import Counter

# ===================== PARAMETERS =====================
# Define base directory relative to this file
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "..", "data")
DUMP_DIR = os.path.join(CURRENT_DIR, "..", "dump")

# CSV file names
TRAIN_PROMPTS_FILE = os.path.join(DATA_DIR, "train_prompts.csv")
TRAIN_RESPONSES_FILE = os.path.join(DATA_DIR, "train_responses.csv")
DEV_PROMPTS_FILE = os.path.join(DATA_DIR, "dev_prompts.csv")
DEV_RESPONSES_FILE = os.path.join(DATA_DIR, "dev_responses.csv")
TEST_PROMPTS_FILE = os.path.join(DATA_DIR, "test_prompts.csv")

# Output file names
DEV_OUTPUT_FILE = os.path.join(DUMP_DIR, "track1_dev.csv")
TEST_OUTPUT_FILE = os.path.join(DUMP_DIR, "track1_test.csv")
# ======================================================

def preprocess_text(text):
    """Basic preprocessing to preserve key terms"""
    if text is None or pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = str(text).lower()
    
    # Replace special characters with space
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def combine_prompt_features(prompt):
    """Extract key features from prompts"""
    words = prompt.lower().split()
    
    # Extract question words that often indicate intent
    question_indicators = [word for word in words if word in {
        'what', 'how', 'why', 'when', 'where', 'who', 'which', 
        'can', 'could', 'would', 'should', 'is', 'are', 'do', 'does'
    }]
    
    # Extract entities (capitalized words often indicate entities)
    original_words = prompt.split()
    entities = [word for word in original_words if word[0].isupper()] if len(original_words) > 0 else []
    
    # Combine features
    features = f"{prompt.lower()} {' '.join(question_indicators)} {' '.join(entities)}"
    return features

def load_data(use_dev=False):
    """Load data based on evaluation mode"""
    print("Loading data...")
    
    # Load train prompts and responses
    train_prompts = pd.read_csv(TRAIN_PROMPTS_FILE)
    train_responses = pd.read_csv(TRAIN_RESPONSES_FILE)
    
    # Merge to get both prompts and responses for training data
    train_data = pd.merge(train_prompts, train_responses, on="conversation_id")
    
    # Make sure output directory exists
    os.makedirs(DUMP_DIR, exist_ok=True)
    
    if use_dev:
        # For dev evaluation, use train data as candidates
        query_prompts = pd.read_csv(DEV_PROMPTS_FILE)
        output_file = DEV_OUTPUT_FILE
        print(f"Dev evaluation mode: {len(train_data)} candidates (train), {len(query_prompts)} queries (dev)")
    else:
        # For test submission, use train + dev as candidates
        dev_prompts = pd.read_csv(DEV_PROMPTS_FILE)
        dev_responses = pd.read_csv(DEV_RESPONSES_FILE)
        dev_data = pd.merge(dev_prompts, dev_responses, on="conversation_id")
        
        # Combine train and dev data
        train_data = pd.concat([train_data, dev_data], ignore_index=True)
        
        # Load test prompts
        query_prompts = pd.read_csv(TEST_PROMPTS_FILE)
        output_file = TEST_OUTPUT_FILE
        print(f"Test mode: {len(train_data)} candidates (train+dev), {len(query_prompts)} queries (test)")
    
    return train_data, query_prompts, output_file

def analyze_responses(df):
    """Analyze the distribution of response patterns"""
    # Count common first words in responses
    first_words = df['model_response'].apply(lambda x: str(x).split()[0].lower() if pd.notna(x) and len(str(x).split()) > 0 else "").value_counts()
    print(f"Top 10 response first words: {first_words.head(10)}")
    
    # Count response length distribution
    df['response_length'] = df['model_response'].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)
    length_stats = df['response_length'].describe()
    print(f"Response length stats: {length_stats}")

def create_count_vectorizer(train_data):
    """Create count vectorizer based on training data"""
    # Extract unique tokens from prompts
    preprocessed_prompts = [preprocess_text(prompt) for prompt in train_data['user_prompt']]
    
    # Create and fit count vectorizer
    count_vectorizer = CountVectorizer(
        ngram_range=(1, 2),  # Use unigrams and bigrams
        max_features=8000,   # Limit vocabulary size
        min_df=2,            # Ignore terms in fewer than 2 documents
        binary=True          # Binary feature presence (1/0 instead of counts)
    )
    count_vectorizer.fit(preprocessed_prompts)
    
    return count_vectorizer

def create_tfidf_vectorizer(train_data):
    """Create TF-IDF vectorizer based on training data"""
    # Extract unique tokens from prompts
    preprocessed_prompts = [preprocess_text(prompt) for prompt in train_data['user_prompt']]
    
    # Create and fit TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),  # Use unigrams, bigrams, and trigrams
        max_features=12000,  # Limit vocabulary size
        min_df=2,            # Ignore terms in fewer than 2 documents
        max_df=0.85,         # Ignore terms in more than 85% of documents
        sublinear_tf=True    # Apply sublinear scaling
    )
    tfidf_vectorizer.fit(preprocessed_prompts)
    
    return tfidf_vectorizer

def apply_vectors_and_find_matches(train_data, query_prompts, count_vectorizer, tfidf_vectorizer):
    """Apply vectorization and find best matches using ensemble approach"""
    print("Vectorizing prompts...")
    
    # Preprocess all prompts
    train_preprocessed = [preprocess_text(text) for text in train_data['user_prompt']]
    query_preprocessed = [preprocess_text(text) for text in query_prompts['user_prompt']]
    
    # Apply count vectorization
    train_count_vectors = count_vectorizer.transform(train_preprocessed)
    query_count_vectors = count_vectorizer.transform(query_preprocessed)
    
    # Apply TF-IDF vectorization
    train_tfidf_vectors = tfidf_vectorizer.transform(train_preprocessed)
    query_tfidf_vectors = tfidf_vectorizer.transform(query_preprocessed)
    
    # Apply dimensionality reduction to TF-IDF vectors
    print("Applying dimensionality reduction...")
    n_components = min(300, min(train_tfidf_vectors.shape) - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    train_tfidf_reduced = svd.fit_transform(train_tfidf_vectors)
    query_tfidf_reduced = svd.transform(query_tfidf_vectors)
    
    # Process in batches
    print("Finding best matches...")
    batch_size = 500
    num_queries = len(query_prompts)
    results = []
    
    for i in range(0, num_queries, batch_size):
        end_idx = min(i + batch_size, num_queries)
        print(f"Processing queries {i+1}-{end_idx} of {num_queries}...")
        
        # Get batch slices
        query_count_batch = query_count_vectors[i:end_idx]
        query_tfidf_batch = query_tfidf_reduced[i:end_idx]
        
        # Compute similarities for count vectors
        count_similarities = cosine_similarity(query_count_batch, train_count_vectors)
        
        # Compute similarities for TF-IDF reduced vectors
        tfidf_similarities = cosine_similarity(query_tfidf_batch, train_tfidf_reduced)
        
        # Ensemble approach: combine similarity scores
        # Weight TF-IDF higher as it typically performs better for semantic similarity
        combined_similarities = 0.3 * count_similarities + 0.7 * tfidf_similarities
        
        # Find best matches
        best_indices = np.argmax(combined_similarities, axis=1)
        
        # Store results
        for j, idx in enumerate(best_indices):
            query_idx = i + j
            if query_idx < len(query_prompts):
                results.append({
                    'conversation_id': query_prompts.iloc[query_idx]['conversation_id'],
                    'response_id': train_data.iloc[idx]['conversation_id']
                })
    
    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser(description="Track 1: Discrete Text Representation with Ensemble Approach")
    parser.add_argument("--use_dev", action="store_true", help="Use dev set for evaluation")
    args = parser.parse_args()
    
    # Load appropriate data
    train_data, query_prompts, output_file = load_data(use_dev=args.use_dev)
    
    # Analyze training data responses (for insights)
    analyze_responses(train_data)
    
    # Create vectorizers
    count_vectorizer = create_count_vectorizer(train_data)
    tfidf_vectorizer = create_tfidf_vectorizer(train_data)
    
    # Apply vectorization and find matches
    retrieval_df = apply_vectors_and_find_matches(
        train_data, query_prompts, count_vectorizer, tfidf_vectorizer
    )
    
    # Save results
    retrieval_df.to_csv(output_file, index=False)
    print(f"Saved retrieval results to {output_file}")

if __name__ == "__main__":
    main()