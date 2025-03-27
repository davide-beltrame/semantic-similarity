#!/usr/bin/env python3
import os
import argparse
import json
import numpy as np
import pandas as pd
import re
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# ===================== PARAMETERS =====================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "..", "data")
DUMP_DIR = os.path.join(CURRENT_DIR, "..", "dump")
RESULTS_DIR = os.path.join(CURRENT_DIR, "..", "results")
os.makedirs(DUMP_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Response filtering
MIN_RESPONSE_LENGTH = 3  # Min number of characters for valid responses

# Grid search configurations
VECTORIZERS = [
    "tfidf_char",      # Character TF-IDF (baseline)
    "tfidf_word",      # Word TF-IDF
    "count_char",      # Character CountVectorizer
    "count_word",      # Word CountVectorizer 
    "tfidf_char_word"  # Combination of character and word
]

# TF-IDF/Count Vectorizer parameters
ANALYZER_OPTIONS = {
    "char": ["char"],
    "word": ["word"],
    "char_word": ["char", "word"]  # Will be used to create combined vectorizers
}

# N-gram range options
NGRAM_RANGES = {
    "char": [(2, 3), (2, 4), (2, 5), (2, 6), (3, 5), (3, 6)],
    "word": [(1, 1), (1, 2), (1, 3), (2, 3)]
}

# Min document frequency options
MIN_DF_OPTIONS = [1, 2, 3, 5]

# Max document frequency options
MAX_DF_OPTIONS = [0.8, 0.9, 0.95, 0.99]

# SVD dimensions (for dimensionality reduction)
SVD_DIMENSIONS = [None, 100, 300, 500]  # None means no SVD

# Results tracking
GRID_SEARCH_RESULTS_FILE = os.path.join(RESULTS_DIR, "track1_grid_search_results.json")
# ======================================================

def preprocess_text(text, add_boundaries=True):
    """
    Preprocess text for vectorization
    - Lowercase and normalize whitespace
    - Optionally add space boundaries for character n-grams
    """
    if pd.isna(text):
        return ""
    text = str(text).lower().strip()
    text = re.sub(r'\s+', ' ', text)
    
    if add_boundaries:
        return ' ' + text + ' '
    return text

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

def create_vectorizer(vectorizer_type, analyzer, ngram_range, min_df, max_df, max_features=100000):
    """
    Create and configure a vectorizer based on specified parameters
    """
    common_params = {
        'ngram_range': ngram_range,
        'min_df': min_df,
        'max_df': max_df,
        'analyzer': analyzer,
        'max_features': max_features
    }
    
    if 'tfidf' in vectorizer_type:
        return TfidfVectorizer(
            use_idf=True,
            norm='l2',
            sublinear_tf=True,
            **common_params
        )
    else:  # Count vectorizer
        return CountVectorizer(
            binary=False,  # Use term frequency
            **common_params
        )

def apply_svd(matrix, n_components):
    """
    Apply dimensionality reduction using SVD if n_components is not None
    """
    if n_components is None:
        return matrix
    
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    return svd.fit_transform(matrix)

def combine_matrices(matrix1, matrix2, weight1=0.5, weight2=0.5):
    """
    Combine two similarity matrices with weights
    """
    # Normalize both matrices to [0, 1] range if not already
    max1 = np.max(matrix1)
    max2 = np.max(matrix2)
    
    if max1 > 0:
        matrix1 = matrix1 / max1
    if max2 > 0:
        matrix2 = matrix2 / max2
    
    # Combine with weights
    return (weight1 * matrix1) + (weight2 * matrix2)

def evaluate_configuration(df_train_prompts, df_train_responses, df_dev_prompts, df_dev_responses, config):
    """
    Evaluate a specific configuration and return the BLEU score
    """
    vectorizer_type = config['vectorizer_type']
    analyzer = config['analyzer']
    ngram_range = tuple(config['ngram_range'])
    min_df = config['min_df']
    max_df = config['max_df']
    svd_dim = config['svd_dim']
    
    # Process texts based on analyzer type
    add_boundaries = 'char' in analyzer
    df_train_prompts['processed'] = df_train_prompts['user_prompt'].apply(lambda x: preprocess_text(x, add_boundaries))
    df_dev_prompts['processed'] = df_dev_prompts['user_prompt'].apply(lambda x: preprocess_text(x, add_boundaries))
    
    # For combined char + word, we create two separate vectorizers
    if vectorizer_type == 'tfidf_char_word':
        # Character vectorizer
        char_vectorizer = create_vectorizer('tfidf_char', 'char', ngram_range, min_df, max_df)
        char_train = char_vectorizer.fit_transform(df_train_prompts['processed'])
        char_dev = char_vectorizer.transform(df_dev_prompts['processed'])
        
        # Word vectorizer
        word_vectorizer = create_vectorizer('tfidf_word', 'word', (1, 3), min_df, max_df)
        word_train = word_vectorizer.fit_transform(df_train_prompts['processed'])
        word_dev = word_vectorizer.transform(df_dev_prompts['processed'])
        
        # Apply SVD if requested
        if svd_dim is not None:
            char_train = apply_svd(char_train, svd_dim)
            char_dev = apply_svd(char_dev, svd_dim)
            word_train = apply_svd(word_train, svd_dim)
            word_dev = apply_svd(word_dev, svd_dim)
        
        # Compute similarity matrices
        char_similarity = cosine_similarity(char_dev, char_train)
        word_similarity = cosine_similarity(word_dev, word_train)
        
        # Combine matrices with weights (could be tuned further)
        similarity_matrix = combine_matrices(char_similarity, word_similarity, 0.6, 0.4)
    else:
        # Single vectorizer approach
        vectorizer = create_vectorizer(vectorizer_type, analyzer, ngram_range, min_df, max_df)
        tfidf_train = vectorizer.fit_transform(df_train_prompts['processed'])
        tfidf_dev = vectorizer.transform(df_dev_prompts['processed'])
        
        # Apply SVD if requested
        if svd_dim is not None:
            tfidf_train = apply_svd(tfidf_train, svd_dim)
            tfidf_dev = apply_svd(tfidf_dev, svd_dim)
        
        # Compute similarity matrix
        similarity_matrix = cosine_similarity(tfidf_dev, tfidf_train)
    
    # Find best matches
    best_indices = np.argmax(similarity_matrix, axis=1)
    best_train_ids = df_train_prompts.iloc[best_indices]['conversation_id'].values
    
    # Build response mappings
    train_response_map = dict(zip(df_train_responses['conversation_id'], df_train_responses['model_response']))
    dev_response_map = dict(zip(df_dev_responses['conversation_id'], df_dev_responses['model_response']))
    
    # Calculate BLEU scores
    smoothing = SmoothingFunction().method3
    bleu_scores = []
    
    for i, dev_conv_id in enumerate(df_dev_prompts['conversation_id']):
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
    return avg_bleu

def run_grid_search(df_train_prompts, df_train_responses, df_dev_prompts, df_dev_responses):
    """
    Run grid search over all configurations
    """
    results = []
    best_bleu = 0
    best_config = None
    
    # Calculate total number of configurations
    total_configs = 0
    for vectorizer_type in VECTORIZERS:
        if vectorizer_type == 'tfidf_char_word':
            # For combined, we don't iterate over analyzer
            for ngram_range in NGRAM_RANGES['char']:
                for min_df in MIN_DF_OPTIONS:
                    for max_df in MAX_DF_OPTIONS:
                        for svd_dim in SVD_DIMENSIONS:
                            total_configs += 1
        else:
            analyzer_type = vectorizer_type.split('_')[1]  # Extract char or word
            for ngram_range in NGRAM_RANGES[analyzer_type]:
                for min_df in MIN_DF_OPTIONS:
                    for max_df in MAX_DF_OPTIONS:
                        for svd_dim in SVD_DIMENSIONS:
                            total_configs += 1
    
    print(f"Starting grid search with {total_configs} configurations...")
    config_count = 0
    
    # Create progress bar
    progress_bar = tqdm(total=total_configs, desc="Evaluating configurations")
    
    # Grid search over vectorizer types
    for vectorizer_type in VECTORIZERS:
        if vectorizer_type == 'tfidf_char_word':
            # For combined, we don't iterate over analyzer
            for ngram_range in NGRAM_RANGES['char']:
                for min_df in MIN_DF_OPTIONS:
                    for max_df in MAX_DF_OPTIONS:
                        for svd_dim in SVD_DIMENSIONS:
                            config = {
                                'vectorizer_type': vectorizer_type,
                                'analyzer': 'char_word',  # Not actually used but for record-keeping
                                'ngram_range': ngram_range,
                                'min_df': min_df,
                                'max_df': max_df,
                                'svd_dim': svd_dim
                            }
                            
                            try:
                                bleu_score = evaluate_configuration(
                                    df_train_prompts.copy(), df_train_responses,
                                    df_dev_prompts.copy(), df_dev_responses,
                                    config
                                )
                                
                                # Record result
                                config_result = config.copy()
                                config_result['bleu_score'] = float(bleu_score)
                                results.append(config_result)
                                
                                # Check if this is the best configuration
                                if bleu_score > best_bleu:
                                    best_bleu = bleu_score
                                    best_config = config.copy()
                                    tqdm.write(f"New best: {best_config}, BLEU: {best_bleu:.5f}")
                                
                                # Save intermediate results
                                with open(GRID_SEARCH_RESULTS_FILE, 'w') as f:
                                    json.dump({
                                        'results': results,
                                        'best_config': best_config,
                                        'best_bleu': float(best_bleu)
                                    }, f, indent=2)
                            
                            except Exception as e:
                                tqdm.write(f"Error evaluating {config}: {e}")
                            
                            config_count += 1
                            progress_bar.update(1)
        else:
            analyzer_type = vectorizer_type.split('_')[1]  # Extract char or word
            for ngram_range in NGRAM_RANGES[analyzer_type]:
                for min_df in MIN_DF_OPTIONS:
                    for max_df in MAX_DF_OPTIONS:
                        for svd_dim in SVD_DIMENSIONS:
                            config = {
                                'vectorizer_type': vectorizer_type,
                                'analyzer': analyzer_type,
                                'ngram_range': ngram_range,
                                'min_df': min_df,
                                'max_df': max_df,
                                'svd_dim': svd_dim
                            }
                            
                            try:
                                bleu_score = evaluate_configuration(
                                    df_train_prompts.copy(), df_train_responses,
                                    df_dev_prompts.copy(), df_dev_responses,
                                    config
                                )
                                
                                # Record result
                                config_result = config.copy()
                                config_result['bleu_score'] = float(bleu_score)
                                results.append(config_result)
                                
                                # Check if this is the best configuration
                                if bleu_score > best_bleu:
                                    best_bleu = bleu_score
                                    best_config = config.copy()
                                    tqdm.write(f"New best: {best_config}, BLEU: {best_bleu:.5f}")
                                
                                # Save intermediate results
                                with open(GRID_SEARCH_RESULTS_FILE, 'w') as f:
                                    json.dump({
                                        'results': results,
                                        'best_config': best_config,
                                        'best_bleu': float(best_bleu)
                                    }, f, indent=2)
                            
                            except Exception as e:
                                tqdm.write(f"Error evaluating {config}: {e}")
                            
                            config_count += 1
                            progress_bar.update(1)
    
    progress_bar.close()
    
    # Final results
    print("\n==== Grid Search Complete ====")
    print(f"Best configuration:")
    print(json.dumps(best_config, indent=2))
    print(f"Best BLEU score: {best_bleu:.5f}")
    
    # Return best configuration
    return best_config, best_bleu

def apply_best_config_to_test(df_train_prompts, df_train_responses, df_dev_prompts, df_dev_responses, df_test_prompts, best_config):
    """
    Apply the best configuration to generate test predictions
    """
    print("\nApplying best configuration to test set...")
    
    vectorizer_type = best_config['vectorizer_type']
    analyzer = best_config['analyzer']
    ngram_range = tuple(best_config['ngram_range'])
    min_df = best_config['min_df']
    max_df = best_config['max_df']
    svd_dim = best_config['svd_dim']
    
    # Create combined candidate pool for test prediction
    candidate_prompts = pd.concat([df_train_prompts, df_dev_prompts], ignore_index=True)
    
    # Process all prompts
    add_boundaries = 'char' in analyzer
    candidate_prompts['processed'] = candidate_prompts['user_prompt'].apply(lambda x: preprocess_text(x, add_boundaries))
    df_test_prompts['processed'] = df_test_prompts['user_prompt'].apply(lambda x: preprocess_text(x, add_boundaries))
    
    # For combined char + word, we create two separate vectorizers
    if vectorizer_type == 'tfidf_char_word':
        # Character vectorizer
        char_vectorizer = create_vectorizer('tfidf_char', 'char', ngram_range, min_df, max_df)
        char_candidates = char_vectorizer.fit_transform(candidate_prompts['processed'])
        char_test = char_vectorizer.transform(df_test_prompts['processed'])
        
        # Word vectorizer
        word_vectorizer = create_vectorizer('tfidf_word', 'word', (1, 3), min_df, max_df)
        word_candidates = word_vectorizer.fit_transform(candidate_prompts['processed'])
        word_test = word_vectorizer.transform(df_test_prompts['processed'])
        
        # Apply SVD if requested
        if svd_dim is not None:
            char_candidates = apply_svd(char_candidates, svd_dim)
            char_test = apply_svd(char_test, svd_dim)
            word_candidates = apply_svd(word_candidates, svd_dim)
            word_test = apply_svd(word_test, svd_dim)
        
        # Compute similarity matrices
        char_similarity = cosine_similarity(char_test, char_candidates)
        word_similarity = cosine_similarity(word_test, word_candidates)
        
        # Combine matrices with weights (could be tuned further)
        similarity_matrix = combine_matrices(char_similarity, word_similarity, 0.6, 0.4)
    else:
        # Single vectorizer approach
        vectorizer = create_vectorizer(vectorizer_type, analyzer, ngram_range, min_df, max_df)
        candidates_matrix = vectorizer.fit_transform(candidate_prompts['processed'])
        test_matrix = vectorizer.transform(df_test_prompts['processed'])
        
        # Apply SVD if requested
        if svd_dim is not None:
            candidates_matrix = apply_svd(candidates_matrix, svd_dim)
            test_matrix = apply_svd(test_matrix, svd_dim)
        
        # Compute similarity matrix
        similarity_matrix = cosine_similarity(test_matrix, candidates_matrix)
    
    # Find best matches
    best_indices = np.argmax(similarity_matrix, axis=1)
    best_candidate_ids = candidate_prompts.iloc[best_indices]['conversation_id'].values
    
    # Create test results
    test_results_df = pd.DataFrame({
        "conversation_id": df_test_prompts['conversation_id'],
        "response_id": best_candidate_ids
    })
    
    # Save results
    test_output_csv = os.path.join(DUMP_DIR, "track_1_best_config_test.csv")
    test_results_df.to_csv(test_output_csv, index=False)
    print(f"Test predictions saved to: {test_output_csv}")
    
    # Save config information for reproducibility
    config_info = {
        "best_config": best_config,
        "description": "This configuration was found by grid search to have the highest BLEU score on the dev set."
    }
    
    config_output_file = os.path.join(RESULTS_DIR, "track_1_best_config.json")
    with open(config_output_file, 'w') as f:
        json.dump(config_info, f, indent=2)
    
    print(f"Configuration details saved to: {config_output_file}")

def main():
    parser = argparse.ArgumentParser(description="Track 1: Grid Search for Discrete Representations")
    parser.add_argument("--use_saved", action="store_true", 
                        help="Use saved best configuration instead of running grid search")
    parser.add_argument("--no_filter", action="store_true",
                        help="Don't filter invalid responses")
    args = parser.parse_args()
    
    filter_responses = not args.no_filter
    
    # Load data
    print("Loading data...")
    df_train_prompts = pd.read_csv(os.path.join(DATA_DIR, "train_prompts.csv"))
    df_train_responses = pd.read_csv(os.path.join(DATA_DIR, "train_responses.csv"))
    df_dev_prompts = pd.read_csv(os.path.join(DATA_DIR, "dev_prompts.csv"))
    df_dev_responses = pd.read_csv(os.path.join(DATA_DIR, "dev_responses.csv"))
    df_test_prompts = pd.read_csv(os.path.join(DATA_DIR, "test_prompts.csv"))
    
    # Apply response filtering if requested
    if filter_responses:
        df_train_prompts, df_train_responses = filter_invalid_responses(df_train_prompts, df_train_responses)
    
    if args.use_saved and os.path.exists(GRID_SEARCH_RESULTS_FILE):
        # Load best configuration from file
        print(f"Loading saved results from {GRID_SEARCH_RESULTS_FILE}")
        with open(GRID_SEARCH_RESULTS_FILE, 'r') as f:
            saved_results = json.load(f)
            best_config = saved_results.get('best_config', {})
            best_bleu = saved_results.get('best_bleu', 0)
        
        if not best_config:
            print("No valid saved configuration found. Running grid search...")
            best_config, best_bleu = run_grid_search(df_train_prompts, df_train_responses, df_dev_prompts, df_dev_responses)
    else:
        # Run grid search
        best_config, best_bleu = run_grid_search(df_train_prompts, df_train_responses, df_dev_prompts, df_dev_responses)
    
    # Apply best configuration to test set
    apply_best_config_to_test(df_train_prompts, df_train_responses, df_dev_prompts, df_dev_responses, df_test_prompts, best_config)

if __name__ == "__main__":
    main()