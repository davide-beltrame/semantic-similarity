import os
import pandas as pd
import numpy as np
import re
import json
import argparse
from tqdm import tqdm
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from gensim.models import FastText
import multiprocessing
from sentence_transformers import SentenceTransformer

# Response filtering
MIN_RESPONSE_LENGTH = 3  # Min number of characters for valid responses

# ===================== PARAMETERS =====================
# Main model components to combine
COMPONENT_TYPES = [
    "tfidf",           # TF-IDF char n-grams (from Track 1)
    "fasttext",        # FastText embeddings (from Track 2)
    "transformer"      # Sentence Transformer (all-mpnet-base-v2)
]

# Weights for component combinations
# Format: (tfidf_weight, fasttext_weight, transformer_weight)
WEIGHT_COMBINATIONS = [
    (0.0, 0.0, 1.0),    # Pure transformer
    (0.3, 0.3, 0.4),    # Balanced combination
    (0.4, 0.2, 0.4),    # More TF-IDF and transformer
    (0.2, 0.4, 0.4),    # More FastText and transformer
    (0.5, 0.0, 0.5),    # TF-IDF and transformer only
    (0.0, 0.5, 0.5),    # FastText and transformer only
    (0.7, 0.3, 0.0),    # TF-IDF and FastText only (no transformer)
    (0.6, 0.2, 0.2),    # TF-IDF dominant
    (0.2, 0.6, 0.2),    # FastText dominant
]

# TF-IDF Configuration options
TFIDF_CONFIGS = [
    {
        "analyzer": "char",
        "ngram_range": (2, 4),
        "min_df": 2,
        "max_df": 0.8,
        "sublinear_tf": True,
        "use_idf": True,
        "norm": "l2",
        "max_features": 100000
    },
    {
        "analyzer": "char",
        "ngram_range": (2, 5),
        "min_df": 2,
        "max_df": 0.8,
        "sublinear_tf": True,
        "use_idf": True,
        "norm": "l2",
        "max_features": 100000
    }
]

# FastText Configuration options
FASTTEXT_CONFIGS = [
    {
        "vector_size": 300,
        "window": 5,
        "min_count": 3,
        "epochs": 20
    }
]

# Transformer Configuration (using all-mpnet-base-v2)
TRANSFORMER_MODEL = "all-mpnet-base-v2"

# Preprocessing options
PREPROCESSING_OPTIONS = [
    "standard",        # Basic preprocessing
    "with_boundaries"  # Add space boundaries for character n-grams
]

# ======================================================

def preprocess_text(text, method="standard"):
    """
    Preprocess text based on specified method
    """
    if pd.isna(text):
        return ""
        
    text = str(text).lower().strip()
    text = re.sub(r'\s+', ' ', text)
    
    if method == "with_boundaries":
        # Add boundary spaces for character n-grams
        return ' ' + text + ' '
    
    return text

def tokenize(text):
    """
    Simple tokenizer for FastText
    """
    if not text or pd.isna(text):
        return []
    
    # Remove punctuation and replace with spaces
    for punct in string.punctuation:
        text = text.replace(punct, ' ')
    
    # Split on whitespace and filter out empty tokens
    tokens = [token for token in text.split() if token]
    return tokens

def train_fasttext_model(texts, config, model_path):
    """
    Train a FastText model with the given configuration
    """
    if os.path.exists(model_path):
        print(f"Loading existing FastText model from {model_path}")
        return FastText.load(model_path)
    
    print("Training new FastText model...")
    print("Tokenizing texts...")
    tokenized_texts = [tokenize(text) for text in tqdm(texts)]
    
    # Remove empty lists
    tokenized_texts = [tokens for tokens in tokenized_texts if tokens]
    
    # Use as many workers as CPU cores
    workers = max(1, multiprocessing.cpu_count() - 1)
    
    # Train FastText model with configuration
    model = FastText(
        tokenized_texts,
        vector_size=config["vector_size"],
        window=config["window"],
        min_count=config["min_count"],
        epochs=config["epochs"],
        workers=workers
    )
    
    print(f"Saving model to {model_path}")
    model.save(model_path)
    
    return model

def get_sentence_embedding(model, sentence):
    """
    Calculate sentence embedding using FastText model
    """
    if not sentence or sentence.isspace():
        # Return zero vector for empty sentences
        return np.zeros(model.vector_size)
    
    # Tokenize the sentence
    words = tokenize(sentence)
    
    if not words:
        return np.zeros(model.vector_size)
    
    # Get embeddings for all words
    word_vectors = []
    
    for word in words:
        try:
            # Get word vector
            word_vector = model.wv[word]
            word_vectors.append(word_vector)
        except KeyError:
            # Skip words not in vocabulary
            continue
    
    if not word_vectors:
        return np.zeros(model.vector_size)
    
    # Convert to numpy array
    word_vectors = np.array(word_vectors)
    
    # Calculate the mean of word vectors
    sentence_vector = np.mean(word_vectors, axis=0)
    
    # Normalize to unit length
    norm = np.linalg.norm(sentence_vector)
    if norm > 0:
        sentence_vector = sentence_vector / norm
    
    return sentence_vector

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

def normalize_matrix(matrix):
    """
    Normalize similarity matrix to [0, 1] range
    """
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    
    if max_val > min_val:
        return (matrix - min_val) / (max_val - min_val)
    return matrix

def combine_similarity_matrices(matrices, weights):
    """
    Combine multiple similarity matrices using specified weights
    """
    if len(matrices) == 0:
        raise ValueError("At least one similarity matrix is required")
    
    if len(matrices) != len(weights):
        raise ValueError("Number of matrices must match number of weights")
    
    # Check if any weight is zero and skip those matrices
    valid_indices = [i for i, w in enumerate(weights) if w > 0]
    if not valid_indices:
        raise ValueError("At least one weight must be greater than zero")
    
    matrices = [matrices[i] for i in valid_indices]
    weights = [weights[i] for i in valid_indices]
    
    # Normalize the weights to sum to 1
    weight_sum = sum(weights)
    if weight_sum > 0:
        weights = [w / weight_sum for w in weights]
    
    # Normalize each matrix to [0, 1] range
    normalized_matrices = [normalize_matrix(m) for m in matrices]
    
    # Combine matrices with weights
    combined = np.zeros_like(normalized_matrices[0])
    for i, matrix in enumerate(normalized_matrices):
        combined += weights[i] * matrix
    
    return combined

def evaluate_configuration(train_prompts, train_responses, dev_prompts, dev_responses, 
                          all_texts, config, model_dir):
    """
    Evaluate a specific hybrid configuration and return the BLEU score
    """
    components = config["components"]
    weights = config["weights"]
    preprocessing = config["preprocessing"]
    
    similarity_matrices = []
    
    # Process TF-IDF if included
    if "tfidf" in components and weights[components.index("tfidf")] > 0:
        tfidf_config = config["tfidf_config"]
        
        # Preprocess text based on method
        train_processed = train_prompts['user_prompt'].apply(lambda x: preprocess_text(x, preprocessing))
        dev_processed = dev_prompts['user_prompt'].apply(lambda x: preprocess_text(x, preprocessing))
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            analyzer=tfidf_config["analyzer"],
            ngram_range=tfidf_config["ngram_range"],
            min_df=tfidf_config["min_df"],
            max_df=tfidf_config["max_df"],
            sublinear_tf=tfidf_config["sublinear_tf"],
            use_idf=tfidf_config["use_idf"],
            norm=tfidf_config["norm"],
            max_features=tfidf_config["max_features"]
        )
        
        # Fit and transform
        train_tfidf = vectorizer.fit_transform(train_processed)
        dev_tfidf = vectorizer.transform(dev_processed)
        
        # Calculate similarity
        tfidf_similarity = cosine_similarity(dev_tfidf, train_tfidf)
        similarity_matrices.append(tfidf_similarity)
    
    # Process FastText if included
    if "fasttext" in components and weights[components.index("fasttext")] > 0:
        fasttext_config = config["fasttext_config"]
        
        # Create model filename
        ft_filename = f"fasttext_vs{fasttext_config['vector_size']}_win{fasttext_config['window']}_mc{fasttext_config['min_count']}_ep{fasttext_config['epochs']}.bin"
        model_path = os.path.join(model_dir, ft_filename)
        
        # Train or load FastText model
        fasttext_model = train_fasttext_model(all_texts, fasttext_config, model_path)
        
        # Generate embeddings
        print("Generating FastText embeddings...")
        train_embeddings = np.array([
            get_sentence_embedding(fasttext_model, text) 
            for text in tqdm(train_prompts['user_prompt'])
        ])
        
        dev_embeddings = np.array([
            get_sentence_embedding(fasttext_model, text) 
            for text in tqdm(dev_prompts['user_prompt'])
        ])
        
        # Calculate similarity
        fasttext_similarity = cosine_similarity(dev_embeddings, train_embeddings)
        similarity_matrices.append(fasttext_similarity)
    
    # Process Transformer if included
    if "transformer" in components and weights[components.index("transformer")] > 0:
        # Load Sentence Transformer model
        print(f"Loading Sentence Transformer model: {TRANSFORMER_MODEL}")
        transformer_model = SentenceTransformer(TRANSFORMER_MODEL)
        
        # Generate embeddings
        print("Generating Transformer embeddings...")
        train_embeddings = transformer_model.encode(train_prompts['user_prompt'].tolist(), 
                                              show_progress_bar=True)
        
        dev_embeddings = transformer_model.encode(dev_prompts['user_prompt'].tolist(), 
                                            show_progress_bar=True)
        
        # Calculate similarity
        transformer_similarity = cosine_similarity(dev_embeddings, train_embeddings)
        similarity_matrices.append(transformer_similarity)
    
    # Combine similarity matrices
    print("Combining similarity matrices...")
    active_indices = [i for i, comp in enumerate(components) if weights[i] > 0]
    active_matrices = similarity_matrices
    active_weights = [weights[i] for i in active_indices]
    
    combined_similarity = combine_similarity_matrices(active_matrices, active_weights)
    
    # Find best matches
    best_indices = np.argmax(combined_similarity, axis=1)
    best_train_ids = train_prompts.iloc[best_indices]['conversation_id'].values
    
    # Build response mappings
    train_response_map = dict(zip(train_responses['conversation_id'], train_responses['model_response']))
    dev_response_map = dict(zip(dev_responses['conversation_id'], dev_responses['model_response']))
    
    # Compute BLEU scores
    print("Computing BLEU scores...")
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

def run_hybrid_grid_search(train_prompts, train_responses, dev_prompts, dev_responses, 
                         results_file, model_dir):
    """
    Run grid search over hybrid configurations combining TF-IDF, FastText, and Transformer
    """
    # Combine all texts for training FastText models
    all_texts = pd.concat([
        train_prompts['user_prompt'],
        dev_prompts['user_prompt']
    ]).tolist()
    
    # Add responses to the training corpus
    all_responses = pd.concat([
        train_responses['model_response'],
        dev_responses['model_response']
    ]).dropna().tolist()
    
    all_texts.extend(all_responses)
    
    # Create all configurations to test
    print("Generating configurations to test...")
    configs = []
    
    for weight_combo in WEIGHT_COMBINATIONS:
        for tfidf_config in TFIDF_CONFIGS:
            for fasttext_config in FASTTEXT_CONFIGS:
                for preprocessing in PREPROCESSING_OPTIONS:
                    # Create component list based on weights
                    components = []
                    weights = []
                    
                    # Add components with non-zero weights
                    if weight_combo[0] > 0:
                        components.append("tfidf")
                        weights.append(weight_combo[0])
                    
                    if weight_combo[1] > 0:
                        components.append("fasttext")
                        weights.append(weight_combo[1])
                    
                    if weight_combo[2] > 0:
                        components.append("transformer")
                        weights.append(weight_combo[2])
                    
                    # Skip configurations with no components
                    if not components:
                        continue
                    
                    configs.append({
                        "components": components,
                        "weights": weights,
                        "tfidf_config": tfidf_config,
                        "fasttext_config": fasttext_config,
                        "preprocessing": preprocessing
                    })
    
    print(f"Generated {len(configs)} configurations to test")
    
    # Track results
    results = []
    best_bleu = 0
    best_config = None
    
    # Create progress bar
    progress_bar = tqdm(total=len(configs), desc="Evaluating configurations")
    
    # Test each configuration
    for config in configs:
        try:
            # Generate a human-readable description of the configuration
            comp_desc = []
            for i, comp in enumerate(config["components"]):
                comp_desc.append(f"{comp}={config['weights'][i]:.1f}")
            
            config_str = f"{' + '.join(comp_desc)}, Preprocessing: {config['preprocessing']}"
            tqdm.write(f"\nEvaluating: {config_str}")
            
            bleu_score = evaluate_configuration(
                train_prompts, train_responses,
                dev_prompts, dev_responses,
                all_texts, config, model_dir
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
    
    # Final results
    print("\n==== Grid Search Complete ====")
    print(f"Best configuration:")
    print(json.dumps(best_config, indent=2))
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
                            test_prompts, all_texts, best_config, model_dir, output_dir):
    """
    Apply the best configuration to generate test predictions
    """
    print("\nApplying best configuration to test set...")
    
    # Create combined candidate pool for testing
    candidate_prompts = pd.concat([train_prompts, dev_prompts], ignore_index=True)
    candidate_responses = pd.concat([train_responses, dev_responses], ignore_index=True)
    
    components = best_config["components"]
    weights = best_config["weights"]
    preprocessing = best_config["preprocessing"]
    
    similarity_matrices = []
    
    # Process TF-IDF if included
    if "tfidf" in components and weights[components.index("tfidf")] > 0:
        tfidf_config = best_config["tfidf_config"]
        
        # Preprocess text based on method
        candidate_processed = candidate_prompts['user_prompt'].apply(lambda x: preprocess_text(x, preprocessing))
        test_processed = test_prompts['user_prompt'].apply(lambda x: preprocess_text(x, preprocessing))
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            analyzer=tfidf_config["analyzer"],
            ngram_range=tfidf_config["ngram_range"],
            min_df=tfidf_config["min_df"],
            max_df=tfidf_config["max_df"],
            sublinear_tf=tfidf_config["sublinear_tf"],
            use_idf=tfidf_config["use_idf"],
            norm=tfidf_config["norm"],
            max_features=tfidf_config["max_features"]
        )
        
        # Fit and transform
        candidate_tfidf = vectorizer.fit_transform(candidate_processed)
        test_tfidf = vectorizer.transform(test_processed)
        
        # Calculate similarity
        tfidf_similarity = cosine_similarity(test_tfidf, candidate_tfidf)
        similarity_matrices.append(tfidf_similarity)
    
    # Process FastText if included
    if "fasttext" in components and weights[components.index("fasttext")] > 0:
        fasttext_config = best_config["fasttext_config"]
        
        # Create model filename
        ft_filename = f"fasttext_vs{fasttext_config['vector_size']}_win{fasttext_config['window']}_mc{fasttext_config['min_count']}_ep{fasttext_config['epochs']}.bin"
        model_path = os.path.join(model_dir, ft_filename)
        
        # Train or load FastText model
        fasttext_model = train_fasttext_model(all_texts, fasttext_config, model_path)
        
        # Generate embeddings
        print("Generating FastText embeddings...")
        candidate_embeddings = np.array([
            get_sentence_embedding(fasttext_model, text) 
            for text in tqdm(candidate_prompts['user_prompt'])
        ])
        
        test_embeddings = np.array([
            get_sentence_embedding(fasttext_model, text) 
            for text in tqdm(test_prompts['user_prompt'])
        ])
        
        # Calculate similarity
        fasttext_similarity = cosine_similarity(test_embeddings, candidate_embeddings)
        similarity_matrices.append(fasttext_similarity)
    
    # Process Transformer if included
    if "transformer" in components and weights[components.index("transformer")] > 0:
        # Load Sentence Transformer model
        print(f"Loading Sentence Transformer model: {TRANSFORMER_MODEL}")
        transformer_model = SentenceTransformer(TRANSFORMER_MODEL)
        
        # Generate embeddings
        print("Generating Transformer embeddings...")
        candidate_embeddings = transformer_model.encode(candidate_prompts['user_prompt'].tolist(), 
                                                 show_progress_bar=True)
        
        test_embeddings = transformer_model.encode(test_prompts['user_prompt'].tolist(), 
                                             show_progress_bar=True)
        
        # Calculate similarity
        transformer_similarity = cosine_similarity(test_embeddings, candidate_embeddings)
        similarity_matrices.append(transformer_similarity)
    
    # Combine similarity matrices
    print("Combining similarity matrices...")
    active_indices = [i for i, comp in enumerate(components) if weights[i] > 0]
    active_matrices = similarity_matrices
    active_weights = [weights[i] for i in active_indices]
    
    combined_similarity = combine_similarity_matrices(active_matrices, active_weights)
    
    # Find best matches
    best_indices = np.argmax(combined_similarity, axis=1)
    best_candidate_ids = candidate_prompts.iloc[best_indices]['conversation_id'].values
    
    # Create test submission file
    test_results_df = pd.DataFrame({
        "conversation_id": test_prompts['conversation_id'],
        "response_id": best_candidate_ids
    })
    
    # Save test predictions
    test_output_csv = os.path.join(output_dir, "track_3_test.csv")
    test_results_df.to_csv(test_output_csv, index=False)
    
    print(f"Test predictions saved to: {test_output_csv}")
    
    # Save best config info
    config_info = {
        "best_config": best_config,
        "description": "This hybrid configuration was found by grid search to have the highest BLEU score on the dev set."
    }
    
    config_output_file = os.path.join(output_dir, "track_3_best_config.json")
    with open(config_output_file, 'w') as f:
        json.dump(config_info, f, indent=2)
    
    print(f"Configuration details saved to: {config_output_file}")

def main():
    parser = argparse.ArgumentParser(description="Track 3: Hybrid Model Grid Search")
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
    model_dir = os.path.join(code_dir, "../models")
    results_dir = os.path.join(code_dir, "../results")
    
    os.makedirs(dump_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    results_file = os.path.join(results_dir, "track_3_hybrid_grid_search_results.json")
    
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
    
    # Combine all texts for training FastText models
    all_texts = pd.concat([
        df_train_prompts['user_prompt'],
        df_dev_prompts['user_prompt'],
        df_test_prompts['user_prompt']
    ]).tolist()
    
    # Add responses to the training corpus
    all_responses = pd.concat([
        df_train_responses['model_response'],
        df_dev_responses['model_response']
    ]).dropna().tolist()
    
    all_texts.extend(all_responses)
    
    if args.use_saved and os.path.exists(results_file):
        # Load best configuration from file
        print(f"Loading saved results from {results_file}")
        with open(results_file, 'r') as f:
            saved_results = json.load(f)
            best_config = saved_results.get('best_config', {})
            best_bleu = saved_results.get('best_bleu', 0)
        
        if not best_config:
            print("No valid saved configuration found. Running grid search...")
            best_config, best_bleu = run_hybrid_grid_search(
                df_train_prompts, df_train_responses,
                df_dev_prompts, df_dev_responses,
                results_file, model_dir
            )
    else:
        # Run grid search
        best_config, best_bleu = run_hybrid_grid_search(
            df_train_prompts, df_train_responses,
            df_dev_prompts, df_dev_responses,
            results_file, model_dir
        )
    
    # Apply best configuration to test set
    apply_best_config_to_test(
        df_train_prompts, df_train_responses,
        df_dev_prompts, df_dev_responses,
        df_test_prompts, all_texts,
        best_config, model_dir, dump_dir
    )

if __name__ == "__main__":
    main()