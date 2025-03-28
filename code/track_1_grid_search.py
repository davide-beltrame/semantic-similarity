import pandas as pd
import numpy as np
import re
import os
import json
import argparse
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Response filtering
MIN_RESPONSE_LENGTH = 3  # Min number of characters for valid responses

# ===================== PARAMETERS =====================
# Vectorizer types
VECTORIZER_TYPES = [
    "tfidf",   # TF-IDF Vectorizer
    "count",   # Count Vectorizer
    "binary",  # Binary Count Vectorizer
    "hashing"  # Hashing Vectorizer
]

# Analyzer options
ANALYZER_OPTIONS = [
    "char",     # Character n-grams
    "word",     # Word n-grams
    "char_wb"   # Character n-grams at word boundaries
]

# N-gram ranges
NGRAM_RANGES = {
    "char": [(2, 3), (2, 4), (2, 5), (3, 5), (1, 3)],
    "word": [(1, 1), (1, 2), (1, 3)],
    "char_wb": [(2, 4), (2, 5), (3, 5)]
}

# Min document frequency options
MIN_DF_OPTIONS = [1, 2, 3, 5]

# Max document frequency options
MAX_DF_OPTIONS = [0.7, 0.8, 0.9, 1.0]

# Sublinear TF options (for TF-IDF)
SUBLINEAR_TF_OPTIONS = [True, False]

# Norm options
NORM_OPTIONS = ['l1', 'l2', None]

# Weighting options (for TF-IDF)
USE_IDF_OPTIONS = [True, False]

# SVD dimensionality reduction options
SVD_DIMENSIONS = [None, 100, 300]  # None = no dimensionality reduction

# Preprocessing options
PREPROCESSING_OPTIONS = [
    "standard",       # Just lowercase and normalize whitespace
    "with_boundaries" # Add word boundaries for character n-grams
]

# =====================================================

def preprocess_text(text, method="standard"):
    """
    Preprocess text for vectorization
    """
    if pd.isna(text):
        return ""
        
    text = str(text).lower().strip()
    text = re.sub(r'\s+', ' ', text)
    
    if method == "with_boundaries":
        # Add boundary spaces for character n-grams
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

def create_vectorizer(config):
    """
    Create and configure a vectorizer based on specified parameters
    """
    vectorizer_type = config["vectorizer_type"]
    analyzer = config["analyzer"]
    ngram_range = tuple(config["ngram_range"])
    min_df = config["min_df"]
    max_df = config["max_df"]
    
    common_params = {
        'analyzer': analyzer,
        'ngram_range': ngram_range,
        'min_df': min_df,
        'max_df': max_df,
        'max_features': 100000
    }
    
    if vectorizer_type == "tfidf":
        sublinear_tf = config.get("sublinear_tf", True)
        norm = config.get("norm", 'l2')
        use_idf = config.get("use_idf", True)
        
        return TfidfVectorizer(
            sublinear_tf=sublinear_tf,
            norm=norm,
            use_idf=use_idf,
            **common_params
        )
    
    elif vectorizer_type == "count":
        return CountVectorizer(
            binary=False,
            **common_params
        )
    
    elif vectorizer_type == "binary":
        return CountVectorizer(
            binary=True,
            **common_params
        )
    
    elif vectorizer_type == "hashing":
        # Hashing vectorizer doesn't support min_df and max_df
        n_features = min(2**18, common_params.pop('max_features', 2**18))
        common_params.pop('min_df', None)
        common_params.pop('max_df', None)
        
        return HashingVectorizer(
            n_features=n_features,
            norm=config.get("norm", 'l2'),
            binary=False,
            **common_params
        )
    
    else:
        raise ValueError(f"Unknown vectorizer type: {vectorizer_type}")

def apply_svd(matrix, n_components):
    """
    Apply SVD for dimensionality reduction
    """
    if n_components is None:
        return matrix
    
    # Ensure we don't try to reduce to more dimensions than we have
    n_components = min(n_components, matrix.shape[1] - 1, matrix.shape[0] - 1)
    
    if n_components <= 0:
        return matrix
    
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    return svd.fit_transform(matrix)

def evaluate_configuration(train_prompts, train_responses, dev_prompts, dev_responses, config):
    """
    Evaluate a specific configuration and return the BLEU score
    """
    # Get configuration parameters
    preprocessing = config.get("preprocessing", "standard")
    svd_dim = config.get("svd_dim")
    
    # Preprocess prompts
    train_prompts['processed'] = train_prompts['user_prompt'].apply(lambda x: preprocess_text(x, preprocessing))
    dev_prompts['processed'] = dev_prompts['user_prompt'].apply(lambda x: preprocess_text(x, preprocessing))
    
    # Create and fit vectorizer
    vectorizer = create_vectorizer(config)
    train_matrix = vectorizer.fit_transform(train_prompts['processed'])
    dev_matrix = vectorizer.transform(dev_prompts['processed'])
    
    # Apply SVD if specified
    if svd_dim is not None:
        train_matrix = apply_svd(train_matrix, svd_dim)
        dev_matrix = apply_svd(dev_matrix, svd_dim)
    
    # Compute similarity and find best matches
    similarity_matrix = cosine_similarity(dev_matrix, train_matrix)
    best_indices = np.argmax(similarity_matrix, axis=1)
    best_train_ids = train_prompts.iloc[best_indices]['conversation_id'].values
    
    # Build response mappings
    train_response_map = dict(zip(train_responses['conversation_id'], train_responses['model_response']))
    dev_response_map = dict(zip(dev_responses['conversation_id'], dev_responses['model_response']))
    
    # Compute BLEU scores
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
    return avg_bleu

def run_smart_grid_search(train_prompts, train_responses, dev_prompts, dev_responses, results_file):
    """
    Run a smart grid search that efficiently explores the parameter space by:
    1. Testing baseline configurations for each vectorizer type
    2. Focusing exploration on the most promising configurations
    3. Fine-tuning the best performing configurations
    """
    results = []
    best_bleu = 0
    best_config = None
    
    # Phase 1: Evaluate baseline configurations for each vectorizer type
    print("Phase 1: Testing baseline configurations...")
    
    baseline_configs = []
    for vect_type in VECTORIZER_TYPES:
        for analyzer in ANALYZER_OPTIONS:
            ngram_ranges = NGRAM_RANGES.get(analyzer, [(2, 4)])
            for ngram in ngram_ranges[:2]:  # Test just the first two ngram ranges for each analyzer
                baseline_configs.append({
                    "vectorizer_type": vect_type,
                    "analyzer": analyzer,
                    "ngram_range": ngram,
                    "min_df": 2,
                    "max_df": 0.8,
                    "preprocessing": "with_boundaries" if analyzer == "char" else "standard",
                    "sublinear_tf": True,
                    "norm": "l2",
                    "use_idf": True,
                    "svd_dim": None
                })
    
    print(f"Testing {len(baseline_configs)} baseline configurations...")
    progress_bar = tqdm(total=len(baseline_configs), desc="Baseline evaluations")
    
    for config in baseline_configs:
        try:
            config_str = (f"Vectorizer: {config['vectorizer_type']}, Analyzer: {config['analyzer']}, "
                          f"NGram: {config['ngram_range']}, Preprocessing: {config['preprocessing']}")
            
            bleu_score = evaluate_configuration(
                train_prompts.copy(), train_responses,
                dev_prompts.copy(), dev_responses,
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
                tqdm.write(f"New best: {config_str}, BLEU: {best_bleu:.5f}")
                
            # Save intermediate results
            with open(results_file, 'w') as f:
                json.dump({
                    'results': sorted(results, key=lambda x: x['bleu_score'], reverse=True),
                    'best_config': best_config,
                    'best_bleu': float(best_bleu)
                }, f, indent=2)
                
        except Exception as e:
            tqdm.write(f"Error evaluating {config}: {str(e)}")
        
        progress_bar.update(1)
    
    progress_bar.close()
    
    # Phase 2: Focus on the most promising vectorizer types and analyzers
    print("\nPhase 2: Exploring the most promising configurations...")
    
    # Sort results and identify the most promising approaches
    sorted_results = sorted(results, key=lambda x: x['bleu_score'], reverse=True)
    top_configs = sorted_results[:min(5, len(sorted_results))]
    
    # Identify the best vectorizer types and analyzers
    best_vect_types = set([c['vectorizer_type'] for c in top_configs[:3]])
    best_analyzers = set([c['analyzer'] for c in top_configs[:3]])
    
    # Generate configurations focusing on the best approaches
    focused_configs = []
    for vect_type in best_vect_types:
        for analyzer in best_analyzers:
            ngram_ranges = NGRAM_RANGES.get(analyzer, [(2, 4)])
            for ngram in ngram_ranges:
                for min_df in MIN_DF_OPTIONS:
                    for max_df in MAX_DF_OPTIONS:
                        for preprocessing in PREPROCESSING_OPTIONS:
                            # Skip inappropriate preprocessing combinations
                            if analyzer != "char" and preprocessing == "with_boundaries":
                                continue
                                
                            # For TF-IDF, explore more hyperparameters
                            if vect_type == "tfidf":
                                for sublinear_tf in SUBLINEAR_TF_OPTIONS:
                                    for norm in NORM_OPTIONS[:2]:  # Skip None for now
                                        for use_idf in USE_IDF_OPTIONS:
                                            focused_configs.append({
                                                "vectorizer_type": vect_type,
                                                "analyzer": analyzer,
                                                "ngram_range": ngram,
                                                "min_df": min_df,
                                                "max_df": max_df,
                                                "preprocessing": preprocessing,
                                                "sublinear_tf": sublinear_tf,
                                                "norm": norm,
                                                "use_idf": use_idf,
                                                "svd_dim": None
                                            })
                            else:
                                # For other vectorizers, just test basic configurations
                                focused_configs.append({
                                    "vectorizer_type": vect_type,
                                    "analyzer": analyzer,
                                    "ngram_range": ngram,
                                    "min_df": min_df,
                                    "max_df": max_df,
                                    "preprocessing": preprocessing,
                                    "norm": "l2" if vect_type == "hashing" else None,
                                    "svd_dim": None
                                })
    
    # If we have too many configurations, sample a reasonable number
    if len(focused_configs) > 200:
        import random
        random.seed(42)
        focused_configs = random.sample(focused_configs, 200)
    
    print(f"Testing {len(focused_configs)} focused configurations...")
    progress_bar = tqdm(total=len(focused_configs), desc="Focused exploration")
    
    for config in focused_configs:
        try:
            bleu_score = evaluate_configuration(
                train_prompts.copy(), train_responses,
                dev_prompts.copy(), dev_responses,
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
                
                config_str = (f"Vectorizer: {config['vectorizer_type']}, Analyzer: {config['analyzer']}, "
                             f"NGram: {config['ngram_range']}, Preprocessing: {config['preprocessing']}")
                tqdm.write(f"New best: {config_str}, BLEU: {best_bleu:.5f}")
                
                # Save intermediate results after each new best
                with open(results_file, 'w') as f:
                    json.dump({
                        'results': sorted(results, key=lambda x: x['bleu_score'], reverse=True),
                        'best_config': best_config,
                        'best_bleu': float(best_bleu)
                    }, f, indent=2)
                
        except Exception as e:
            tqdm.write(f"Error evaluating {config}: {str(e)}")
        
        progress_bar.update(1)
    
    progress_bar.close()
    
    # Phase 3: Test SVD on the best configurations
    print("\nPhase 3: Testing SVD dimensionality reduction...")
    
    # Test SVD on the top 3 configurations
    top_configs = sorted(results, key=lambda x: x['bleu_score'], reverse=True)[:3]
    svd_configs = []
    
    for base_config in top_configs:
        for svd_dim in SVD_DIMENSIONS:
            if svd_dim is not None:  # Skip None since we already tested it
                svd_config = base_config.copy()
                svd_config['svd_dim'] = svd_dim
                svd_configs.append(svd_config)
    
    print(f"Testing {len(svd_configs)} SVD configurations...")
    progress_bar = tqdm(total=len(svd_configs), desc="SVD testing")
    
    for config in svd_configs:
        try:
            bleu_score = evaluate_configuration(
                train_prompts.copy(), train_responses,
                dev_prompts.copy(), dev_responses,
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
                
                config_str = (f"Vectorizer: {config['vectorizer_type']}, SVD: {config['svd_dim']}, "
                             f"BLEU: {bleu_score:.5f}")
                tqdm.write(f"New best with SVD: {config_str}")
                
        except Exception as e:
            tqdm.write(f"Error evaluating SVD config: {str(e)}")
        
        progress_bar.update(1)
    
    progress_bar.close()
    
    # Final results
    print("\n==== Grid Search Complete ====")
    print(f"Best configuration:")
    for key, value in best_config.items():
        print(f"  {key}: {value}")
    print(f"Best BLEU score: {best_bleu:.5f}")
    
    # Save final results
    with open(results_file, 'w') as f:
        json.dump({
            'results': sorted(results, key=lambda x: x['bleu_score'], reverse=True),
            'best_config': best_config,
            'best_bleu': float(best_bleu)
        }, f, indent=2)
    
    return best_config, best_bleu

def apply_best_config_to_test(train_prompts, train_responses, dev_prompts, dev_responses, 
                            test_prompts, best_config, output_dir):
    """
    Apply the best configuration to generate test predictions
    """
    print("\nApplying best configuration to test set...")
    
    # Create combined candidate pool (train + dev) for testing
    candidate_prompts = pd.concat([train_prompts, dev_prompts], ignore_index=True)
    
    # Preprocess prompts
    preprocessing = best_config.get("preprocessing", "standard")
    candidate_prompts['processed'] = candidate_prompts['user_prompt'].apply(
        lambda x: preprocess_text(x, preprocessing))
    test_prompts['processed'] = test_prompts['user_prompt'].apply(
        lambda x: preprocess_text(x, preprocessing))
    
    # Create and fit vectorizer
    vectorizer = create_vectorizer(best_config)
    candidate_matrix = vectorizer.fit_transform(candidate_prompts['processed'])
    test_matrix = vectorizer.transform(test_prompts['processed'])
    
    # Apply SVD if specified
    svd_dim = best_config.get("svd_dim")
    if svd_dim is not None:
        candidate_matrix = apply_svd(candidate_matrix, svd_dim)
        test_matrix = apply_svd(test_matrix, svd_dim)
    
    # Compute similarity and find best matches
    similarity_matrix = cosine_similarity(test_matrix, candidate_matrix)
    best_indices = np.argmax(similarity_matrix, axis=1)
    best_candidate_ids = candidate_prompts.iloc[best_indices]['conversation_id'].values
    
    # Create test submission file
    test_results_df = pd.DataFrame({
        "conversation_id": test_prompts['conversation_id'],
        "response_id": best_candidate_ids
    })
    
    # Save test predictions
    test_output_csv = os.path.join(output_dir, "track_1_test.csv")
    test_results_df.to_csv(test_output_csv, index=False)
    
    print(f"Test predictions saved to: {test_output_csv}")
    
    # Save best config info
    config_info = {
        "best_config": best_config,
        "description": "This configuration was found by grid search to have the highest BLEU score on the dev set."
    }
    
    config_output_file = os.path.join(output_dir, "track_1_best_config.json")
    with open(config_output_file, 'w') as f:
        json.dump(config_info, f, indent=2)
    
    print(f"Configuration details saved to: {config_output_file}")

def main():
    parser = argparse.ArgumentParser(description="Track 1: Smart Grid Search for Discrete Representations")
    parser.add_argument("--use_saved", action="store_true", 
                        help="Use saved best configuration instead of running grid search")
    parser.add_argument("--no_filter", action="store_true",
                        help="Don't filter invalid responses")
    args = parser.parse_args()
    
    filter_responses = not args.no_filter
    
    # Set up paths
    code_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(code_dir, "../data")
    dump_dir = os.path.join(code_dir, "../dump")
    results_dir = os.path.join(code_dir, "../results")
    
    os.makedirs(dump_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    results_file = os.path.join(results_dir, "track_1_grid_search_results.json")
    
    # Load data
    print("Loading data...")
    df_train_prompts = pd.read_csv(os.path.join(data_dir, "train_prompts.csv"))
    df_train_responses = pd.read_csv(os.path.join(data_dir, "train_responses.csv"))
    df_dev_prompts = pd.read_csv(os.path.join(data_dir, "dev_prompts.csv"))
    df_dev_responses = pd.read_csv(os.path.join(data_dir, "dev_responses.csv"))
    df_test_prompts = pd.read_csv(os.path.join(data_dir, "test_prompts.csv"))
    
    # Apply response filtering if requested
    if filter_responses:
        df_train_prompts, df_train_responses = filter_invalid_responses(df_train_prompts, df_train_responses)
    
    if args.use_saved and os.path.exists(results_file):
        # Load best configuration from file
        print(f"Loading saved results from {results_file}")
        with open(results_file, 'r') as f:
            saved_results = json.load(f)
            best_config = saved_results.get('best_config', {})
            best_bleu = saved_results.get('best_bleu', 0)
        
        if not best_config:
            print("No valid saved configuration found. Running grid search...")
            best_config, best_bleu = run_smart_grid_search(
                df_train_prompts, df_train_responses,
                df_dev_prompts, df_dev_responses,
                results_file
            )
    else:
        # Run grid search
        best_config, best_bleu = run_smart_grid_search(
            df_train_prompts, df_train_responses,
            df_dev_prompts, df_dev_responses,
            results_file
        )
    
    # Apply best configuration to test set
    apply_best_config_to_test(
        df_train_prompts, df_train_responses,
        df_dev_prompts, df_dev_responses,
        df_test_prompts, best_config, dump_dir
    )

if __name__ == "__main__":
    main()