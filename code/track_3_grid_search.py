#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
import re
import json
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# ===================== PARAMETERS =====================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "..", "data")
DUMP_DIR = os.path.join(CURRENT_DIR, "..", "dump")
RESULTS_DIR = os.path.join(CURRENT_DIR, "..", "results")
os.makedirs(DUMP_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# CSV file names
TRAIN_PROMPTS_FILE = os.path.join(DATA_DIR, "train_prompts.csv")
TRAIN_RESPONSES_FILE = os.path.join(DATA_DIR, "train_responses.csv")
DEV_PROMPTS_FILE = os.path.join(DATA_DIR, "dev_prompts.csv")
DEV_RESPONSES_FILE = os.path.join(DATA_DIR, "dev_responses.csv")
TEST_PROMPTS_FILE = os.path.join(DATA_DIR, "test_prompts.csv")

# Grid search parameters
MODELS = [
    "all-mpnet-base-v2",           # Our current best model
    "all-MiniLM-L6-v2",            # Faster, smaller model
    "multi-qa-mpnet-base-dot-v1",  # Specialized for questions
    "paraphrase-multilingual-mpnet-base-v2", # Good for paraphrasing
    "all-distilroberta-v1",        # Good balance of speed and quality
    "all-MiniLM-L12-v2",           # Better than L6, still fast
    "multi-qa-distilbert-cos-v1",  # Optimized for questions, smaller
    "msmarco-distilbert-base-v4",  # Good for information retrieval
    "distiluse-base-multilingual-cased-v1", # Good for multilingual
    "sentence-t5-base",            # T5-based model, good quality
]

PREPROCESSING_METHODS = [
    "basic",        # Just lowercase and normalize whitespace
    "char_bounded", # Add spaces around text (current method)
    "no_punct",     # Remove punctuation
    "simple_tokens", # Simple tokenization without NLTK
]

SIMILARITY_METRICS = [
    "cosine",
    "euclidean",
    "manhattan",
]

# Response filtering
MIN_RESPONSE_LENGTH = 3  # Min number of characters for valid responses

# Results tracking
GRID_SEARCH_RESULTS_FILE = os.path.join(RESULTS_DIR, "grid_search_results.json")
# ======================================================

def preprocess_basic(text):
    """Basic preprocessing: lowercase and normalize whitespace"""
    if pd.isna(text):
        return ""
    text = str(text).lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def preprocess_char_bounded(text):
    """Add spaces around text to capture word boundaries"""
    if pd.isna(text):
        return ""
    text = str(text).lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return ' ' + text + ' '

def preprocess_no_punct(text):
    """Remove punctuation"""
    if pd.isna(text):
        return ""
    text = str(text).lower().strip()
    text = re.sub(r'[^\w\s]', ' ', text)  # Replace punctuation with space
    text = re.sub(r'\s+', ' ', text)
    return text

def preprocess_simple_tokens(text):
    """
    Simple word tokenization without NLTK.
    Removes punctuation, splits on whitespace, and rejoins.
    """
    if pd.isna(text):
        return ""
    text = str(text).lower().strip()
    # Replace punctuation with spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    # Split on whitespace and filter out empty tokens
    tokens = [token for token in text.split() if token]
    return ' '.join(tokens)

def get_preprocessing_function(method_name):
    """Return the preprocessing function based on name"""
    preprocess_functions = {
        "basic": preprocess_basic,
        "char_bounded": preprocess_char_bounded,
        "no_punct": preprocess_no_punct,
        "simple_tokens": preprocess_simple_tokens
    }
    return preprocess_functions.get(method_name, preprocess_basic)

def calculate_similarity(embeddings1, embeddings2, metric):
    """Calculate similarity using different metrics"""
    if metric == "cosine":
        return cosine_similarity(embeddings1, embeddings2)
    elif metric == "euclidean":
        from sklearn.metrics.pairwise import euclidean_distances
        # Convert to similarity (higher is better)
        distances = euclidean_distances(embeddings1, embeddings2)
        # Invert and normalize to [0, 1] range
        max_dist = np.max(distances)
        if max_dist > 0:
            return 1 - (distances / max_dist)
        return distances  # All zeros
    elif metric == "manhattan":
        from sklearn.metrics.pairwise import manhattan_distances
        # Convert to similarity (higher is better)
        distances = manhattan_distances(embeddings1, embeddings2)
        # Invert and normalize to [0, 1] range
        max_dist = np.max(distances)
        if max_dist > 0:
            return 1 - (distances / max_dist)
        return distances  # All zeros
    else:
        # Default to cosine similarity
        return cosine_similarity(embeddings1, embeddings2)

def filter_invalid_responses(train_prompts, train_responses):
    """
    Filter out training examples where the response is too short or invalid.
    Returns filtered dataframes.
    """
    print(f"Filtering invalid responses (min length: {MIN_RESPONSE_LENGTH} chars)...")
    
    # Get valid response IDs
    valid_response_ids = []
    invalid_count = 0
    
    for _, row in train_responses.iterrows():
        response = row['model_response']
        conv_id = row['conversation_id']
        
        if pd.isna(response) or len(str(response).strip()) < MIN_RESPONSE_LENGTH:
            invalid_count += 1
        else:
            valid_response_ids.append(conv_id)
    
    # Filter prompts to only include those with valid responses
    filtered_train_prompts = train_prompts[train_prompts['conversation_id'].isin(valid_response_ids)].copy()
    filtered_train_responses = train_responses[train_responses['conversation_id'].isin(valid_response_ids)].copy()
    
    print(f"Filtered out {invalid_count} invalid responses. Remaining: {len(filtered_train_prompts)} examples")
    
    return filtered_train_prompts, filtered_train_responses

def load_data(filter_responses=True):
    """Load all data sets with optional response filtering"""
    train_prompts = pd.read_csv(TRAIN_PROMPTS_FILE)
    train_responses = pd.read_csv(TRAIN_RESPONSES_FILE)
    dev_prompts = pd.read_csv(DEV_PROMPTS_FILE)
    dev_responses = pd.read_csv(DEV_RESPONSES_FILE)
    test_prompts = pd.read_csv(TEST_PROMPTS_FILE)
    
    # Apply response filtering if requested
    if filter_responses:
        train_prompts, train_responses = filter_invalid_responses(train_prompts, train_responses)
    
    return train_prompts, train_responses, dev_prompts, dev_responses, test_prompts

def run_grid_search(filter_responses=True):
    """
    Run grid search over models, preprocessing methods, and similarity metrics.
    Return the best configuration found.
    """
    print("Starting grid search...")
    
    # Load data
    train_prompts, train_responses, dev_prompts, dev_responses, _ = load_data(filter_responses=filter_responses)
    
    # Initialize results tracking
    results = []
    best_bleu = 0
    best_config = {}
    
    # Grid search over all combinations
    total_configs = len(MODELS) * len(PREPROCESSING_METHODS) * len(SIMILARITY_METRICS)
    config_count = 0
    
    for model_name in MODELS:
        # Load model (time-consuming, only do once per model)
        print(f"Loading model: {model_name}")
        try:
            model = SentenceTransformer(model_name)
            
            for preprocess_method in PREPROCESSING_METHODS:
                preprocess_fn = get_preprocessing_function(preprocess_method)
                
                # Preprocess data
                print(f"Preprocessing with method: {preprocess_method}")
                train_prompts['processed'] = train_prompts['user_prompt'].apply(preprocess_fn)
                dev_prompts['processed'] = dev_prompts['user_prompt'].apply(preprocess_fn)
                
                # Encode texts (time-consuming, only do once per model+preprocess combo)
                print("Encoding train prompts...")
                train_embeddings = model.encode(
                    train_prompts['processed'].tolist(), 
                    show_progress_bar=True,
                    batch_size=32
                )
                
                print("Encoding dev prompts...")
                dev_embeddings = model.encode(
                    dev_prompts['processed'].tolist(), 
                    show_progress_bar=True,
                    batch_size=32
                )
                
                for similarity_metric in SIMILARITY_METRICS:
                    config_count += 1
                    print(f"\nTrying configuration {config_count}/{total_configs}:")
                    print(f"  Model: {model_name}")
                    print(f"  Preprocessing: {preprocess_method}")
                    print(f"  Similarity metric: {similarity_metric}")
                    print(f"  Response filtering: {filter_responses}")
                    
                    # Calculate similarities
                    print("Computing similarities...")
                    similarities = calculate_similarity(dev_embeddings, train_embeddings, similarity_metric)
                    
                    # Exclude self-matches
                    candidate_ids = train_prompts["conversation_id"].tolist()
                    query_ids = dev_prompts["conversation_id"].tolist()
                    for i, qid in enumerate(query_ids):
                        for j, cid in enumerate(candidate_ids):
                            if qid == cid:
                                similarities[i, j] = -float('inf')  # Force non-selection
                    
                    # Get best matches
                    best_indices = np.argmax(similarities, axis=1)
                    best_train_ids = train_prompts.iloc[best_indices]['conversation_id'].values
                    
                    # Build response mappings
                    train_response_map = dict(zip(train_responses['conversation_id'], train_responses['model_response']))
                    dev_response_map = dict(zip(dev_responses['conversation_id'], dev_responses['model_response']))
                    
                    # Calculate BLEU scores
                    print("Calculating BLEU scores...")
                    smoothing = SmoothingFunction().method3
                    bleu_scores = []
                    
                    for i, dev_conv_id in enumerate(dev_prompts['conversation_id']):
                        true_resp = str(dev_response_map.get(dev_conv_id, ""))
                        retrieved_id = best_train_ids[i]
                        pred_resp = str(train_response_map.get(retrieved_id, ""))
                        
                        if not true_resp or not pred_resp:
                            bleu = 0.0
                        else:
                            bleu = sentence_bleu(
                                [true_resp.split()],
                                pred_resp.split(),
                                weights=(0.5, 0.5, 0, 0),
                                smoothing_function=smoothing
                            )
                        bleu_scores.append(bleu)
                    
                    avg_bleu = np.mean(bleu_scores)
                    print(f"  Average BLEU score: {avg_bleu:.5f}")
                    
                    # Record result
                    config_result = {
                        "model": model_name,
                        "preprocessing": preprocess_method,
                        "similarity_metric": similarity_metric,
                        "response_filtering": filter_responses,
                        "bleu_score": float(avg_bleu)
                    }
                    results.append(config_result)
                    
                    # Check if this is the best configuration so far
                    if avg_bleu > best_bleu:
                        best_bleu = avg_bleu
                        best_config = config_result
                        print(f"  New best configuration! BLEU: {best_bleu:.5f}")
                    
                    # Save intermediate results in case of crashes
                    with open(GRID_SEARCH_RESULTS_FILE, 'w') as f:
                        json.dump({
                            "results": results,
                            "best_config": best_config
                        }, f, indent=2)
        
        except Exception as e:
            print(f"Error with model {model_name}: {e}")
            # Continue with next model
            continue
    
    # Final results
    print("\n==== Grid Search Complete ====")
    print(f"Best configuration:")
    print(f"  Model: {best_config.get('model')}")
    print(f"  Preprocessing: {best_config.get('preprocessing')}")
    print(f"  Similarity metric: {best_config.get('similarity_metric')}")
    print(f"  Response filtering: {best_config.get('response_filtering')}")
    print(f"  BLEU score: {best_config.get('bleu_score'):.5f}")
    
    # Return the best configuration
    return best_config

def apply_best_config_to_test(best_config):
    """Apply the best configuration to generate test predictions"""
    print("\nApplying best configuration to test set...")
    
    # Extract configuration parameters
    model_name = best_config.get('model')
    preprocess_method = best_config.get('preprocessing')
    similarity_metric = best_config.get('similarity_metric')
    filter_responses = best_config.get('response_filtering', True)
    
    # Load data
    train_prompts, train_responses, dev_prompts, dev_responses, test_prompts = load_data(filter_responses=filter_responses)
    
    # Concatenate train and dev for the candidate pool
    candidate_prompts = pd.concat([train_prompts, dev_prompts], ignore_index=True)
    
    # For candidates from dev set, we need to check if they have responses
    dev_response_ids = set(dev_responses['conversation_id'])
    
    # Filter out dev prompts without responses if filtering is enabled
    if filter_responses:
        print("Filtering dev prompts for candidate pool...")
        candidate_prompts = candidate_prompts[
            (candidate_prompts['conversation_id'].isin(train_prompts['conversation_id'])) | 
            (candidate_prompts['conversation_id'].isin(dev_response_ids))
        ]
    
    # Apply preprocessing
    preprocess_fn = get_preprocessing_function(preprocess_method)
    candidate_prompts['processed'] = candidate_prompts['user_prompt'].apply(preprocess_fn)
    test_prompts['processed'] = test_prompts['user_prompt'].apply(preprocess_fn)
    
    # Load model
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    
    # Encode texts
    print("Encoding candidate prompts...")
    candidate_embeddings = model.encode(
        candidate_prompts['processed'].tolist(), 
        show_progress_bar=True,
        batch_size=32
    )
    
    print("Encoding test prompts...")
    test_embeddings = model.encode(
        test_prompts['processed'].tolist(), 
        show_progress_bar=True,
        batch_size=32
    )
    
    # Calculate similarities
    print("Computing similarities...")
    similarities = calculate_similarity(test_embeddings, candidate_embeddings, similarity_metric)
    
    # Get best matches
    best_indices = np.argmax(similarities, axis=1)
    best_candidate_ids = candidate_prompts.iloc[best_indices]['conversation_id'].values
    
    # Create submission file
    test_results_df = pd.DataFrame({
        "conversation_id": test_prompts['conversation_id'],
        "response_id": best_candidate_ids
    })
    
    # Save results
    output_file = os.path.join(DUMP_DIR, "track_3_grid_search_test.csv")
    test_results_df.to_csv(output_file, index=False)
    print(f"Test predictions saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Grid search for Track 3 semantic similarity")
    parser.add_argument("--use_saved", action="store_true", 
                        help="Use saved best configuration instead of running grid search")
    parser.add_argument("--no_filter", action="store_true",
                        help="Don't filter invalid responses")
    args = parser.parse_args()
    
    filter_responses = not args.no_filter
    
    if args.use_saved and os.path.exists(GRID_SEARCH_RESULTS_FILE):
        # Load best configuration from file
        print(f"Loading saved results from {GRID_SEARCH_RESULTS_FILE}")
        with open(GRID_SEARCH_RESULTS_FILE, 'r') as f:
            saved_results = json.load(f)
            best_config = saved_results.get('best_config', {})
        
        if not best_config:
            print("No valid saved configuration found. Running grid search...")
            best_config = run_grid_search(filter_responses=filter_responses)
    else:
        # Run grid search
        best_config = run_grid_search(filter_responses=filter_responses)
    
    # Apply best configuration to test set
    apply_best_config_to_test(best_config)

if __name__ == "__main__":
    main()