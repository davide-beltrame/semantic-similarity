import pandas as pd
import numpy as np
import re
import os
import string
import json
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm
from gensim.models import FastText, Word2Vec
import multiprocessing
import argparse

# ===================== PARAMETERS =====================
# Response filtering
MIN_RESPONSE_LENGTH = 3  # Min number of characters for valid responses

# Grid search parameters
MODEL_TYPES = ["fasttext", "word2vec"]
VECTOR_SIZES = [100, 200, 300]
WINDOW_SIZES = [3, 5, 8]
MIN_COUNTS = [2, 3, 5]
EPOCHS = [10, 20]

# Sentence embedding methods
EMBEDDING_METHODS = [
    "mean",        # Simple average of word vectors
    "weighted",    # Weighted average based on inverse frequency
    "max_pooling"  # Max pooling across each dimension
]

# ======================================================

def preprocess_text(text, remove_punct=True, lowercase=True):
    """
    Preprocess text for word embeddings:
    - Convert to string
    - Lowercase (optional)
    - Normalize whitespace
    - Remove punctuation (optional)
    """
    if pd.isna(text):
        return ""
    
    text = str(text)
    
    if lowercase:
        text = text.lower()
    
    # Normalize whitespace
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    
    if remove_punct:
        # Replace punctuation with spaces
        for punct in string.punctuation:
            text = text.replace(punct, ' ')
        
        # Normalize whitespace again after punctuation removal
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
    
    return text

def tokenize(text, remove_punct=True):
    """
    Tokenize text into words.
    """
    if not text or pd.isna(text):
        return []
    
    if remove_punct:
        # Remove punctuation and replace with spaces
        for punct in string.punctuation:
            text = text.replace(punct, ' ')
    
    # Split on whitespace and filter out empty tokens
    tokens = [token for token in text.split() if token]
    return tokens

def calculate_word_weights(all_texts, remove_punct=True):
    """
    Calculate word weights based on inverse document frequency-like metric
    """
    word_counts = {}
    doc_counts = {}
    total_docs = len(all_texts)
    
    # Count words in each document
    for text in all_texts:
        tokens = set(tokenize(text, remove_punct))
        for token in tokens:
            doc_counts[token] = doc_counts.get(token, 0) + 1
    
    # Calculate weights (inverse document frequency)
    word_weights = {}
    for word, count in doc_counts.items():
        word_weights[word] = np.log(total_docs / (1 + count))
    
    return word_weights

def train_model(texts, config, model_dir):
    """
    Train a word embedding model (FastText or Word2Vec) using the specified configuration
    """
    model_type = config["model_type"]
    vector_size = config["vector_size"]
    window = config["window"]
    min_count = config["min_count"]
    epochs = config["epochs"]
    remove_punct = config.get("remove_punct", True)
    
    # Create a unique model filename based on config
    model_filename = f"{model_type}_vs{vector_size}_win{window}_mc{min_count}_ep{epochs}"
    if not remove_punct:
        model_filename += "_withpunct"
    model_path = os.path.join(model_dir, f"{model_filename}.bin")
    
    # Check if model exists
    if os.path.exists(model_path):
        print(f"Loading existing model: {model_filename}")
        if model_type == "fasttext":
            return FastText.load(model_path)
        else:  # word2vec
            return Word2Vec.load(model_path)
    
    print(f"Training new {model_type} model with config: {config}")
    print("Tokenizing texts...")
    tokenized_texts = [tokenize(text, remove_punct) for text in tqdm(texts)]
    
    # Remove empty lists
    tokenized_texts = [tokens for tokens in tokenized_texts if tokens]
    
    # Get max workers
    workers = max(1, multiprocessing.cpu_count() - 1)
    
    # Train model
    if model_type == "fasttext":
        model = FastText(
            tokenized_texts,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            epochs=epochs,
            workers=workers
        )
    else:  # word2vec
        model = Word2Vec(
            tokenized_texts,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            epochs=epochs,
            workers=workers
        )
    
    # Save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    return model

def get_sentence_embedding(model, sentence, method="mean", word_weights=None, remove_punct=True):
    """
    Calculate sentence embedding using different methods:
    - mean: Simple average of word vectors
    - weighted: Weighted average based on provided weights
    - max_pooling: Take the maximum value for each dimension
    """
    model_type = model.__class__.__name__
    vector_size = model.vector_size if hasattr(model, 'vector_size') else model.wv.vector_size
    
    if not sentence or sentence.isspace():
        return np.zeros(vector_size)
    
    # Tokenize the sentence
    words = tokenize(sentence, remove_punct)
    
    if not words:
        return np.zeros(vector_size)
    
    # Get embeddings for all words
    word_vectors = []
    weight_values = []
    
    for word in words:
        try:
            if model_type == 'FastText':
                word_vector = model.wv[word]
            else:  # Word2Vec
                word_vector = model.wv[word]
            
            word_vectors.append(word_vector)
            
            if method == "weighted" and word_weights:
                weight = word_weights.get(word, 1.0)
                weight_values.append(weight)
        except KeyError:
            # Skip words not in vocabulary
            continue
    
    if not word_vectors:
        return np.zeros(vector_size)
    
    # Convert to numpy array
    word_vectors = np.array(word_vectors)
    
    # Calculate embedding based on method
    if method == "mean":
        sentence_vector = np.mean(word_vectors, axis=0)
    
    elif method == "weighted":
        if not weight_values:
            sentence_vector = np.mean(word_vectors, axis=0)
        else:
            weight_values = np.array(weight_values).reshape(-1, 1)
            sentence_vector = np.sum(word_vectors * weight_values, axis=0) / np.sum(weight_values)
    
    elif method == "max_pooling":
        sentence_vector = np.max(word_vectors, axis=0)
    
    else:
        # Default to mean
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

def evaluate_configuration(df_train_prompts, df_train_responses, df_dev_prompts, df_dev_responses, 
                          all_texts, config, model_dir):
    """
    Evaluate a specific model and embedding configuration and return BLEU score
    """
    model_type = config["model_type"]
    embedding_method = config["embedding_method"]
    remove_punct = config.get("remove_punct", True)
    
    # Train or load model
    model = train_model(all_texts, config, model_dir)
    
    # Calculate word weights if using weighted embedding
    word_weights = None
    if embedding_method == "weighted":
        word_weights = calculate_word_weights(all_texts, remove_punct)
    
    # Generate embeddings
    print(f"Generating embeddings with {embedding_method} method...")
    train_embeddings = np.array([
        get_sentence_embedding(model, text, embedding_method, word_weights, remove_punct) 
        for text in tqdm(df_train_prompts['user_prompt'])
    ])
    
    dev_embeddings = np.array([
        get_sentence_embedding(model, text, embedding_method, word_weights, remove_punct) 
        for text in tqdm(df_dev_prompts['user_prompt'])
    ])
    
    # Compute similarity and find best matches
    print("Computing similarity matrix...")
    similarity_matrix = cosine_similarity(dev_embeddings, train_embeddings)
    best_indices = np.argmax(similarity_matrix, axis=1)
    best_train_ids = df_train_prompts.iloc[best_indices]['conversation_id'].values
    
    # Build response mappings
    train_response_map = dict(zip(df_train_responses['conversation_id'], df_train_responses['model_response']))
    dev_response_map = dict(zip(df_dev_responses['conversation_id'], df_dev_responses['model_response']))
    
    # Compute BLEU scores
    print("Computing BLEU scores...")
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

def run_grid_search(df_train_prompts, df_train_responses, df_dev_prompts, df_dev_responses, 
                    results_file, model_dir):
    """
    Run grid search over all configurations
    """
    # Combine all texts for training
    all_texts = pd.concat([
        df_train_prompts['user_prompt'],
        df_dev_prompts['user_prompt']
    ]).tolist()
    
    # Add responses to the training corpus to capture response patterns
    all_responses = pd.concat([
        df_train_responses['model_response'],
        df_dev_responses['model_response']
    ]).dropna().tolist()
    
    all_texts.extend(all_responses)
    
    # Define all configurations to test
    configs = []
    for model_type in MODEL_TYPES:
        for vector_size in VECTOR_SIZES:
            for window in WINDOW_SIZES:
                for min_count in MIN_COUNTS:
                    for epochs in EPOCHS:
                        for emb_method in EMBEDDING_METHODS:
                            for remove_punct in [True, False]:
                                configs.append({
                                    "model_type": model_type,
                                    "vector_size": vector_size,
                                    "window": window,
                                    "min_count": min_count,
                                    "epochs": epochs,
                                    "embedding_method": emb_method,
                                    "remove_punct": remove_punct
                                })
    
    print(f"Starting grid search with {len(configs)} configurations...")
    
    # Track results
    results = []
    best_bleu = 0
    best_config = None
    
    # Create progress bar
    progress_bar = tqdm(total=len(configs), desc="Evaluating configurations")
    
    # Test each configuration
    for config in configs:
        try:
            config_str = f"Model: {config['model_type']}, VS: {config['vector_size']}, Win: {config['window']}, " \
                        f"MinC: {config['min_count']}, Epochs: {config['epochs']}, Emb: {config['embedding_method']}, " \
                        f"RemovePunct: {config['remove_punct']}"
            print(f"\nEvaluating: {config_str}")
            
            bleu_score = evaluate_configuration(
                df_train_prompts, df_train_responses,
                df_dev_prompts, df_dev_responses,
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
            tqdm.write(f"Error evaluating {config}: {e}")
        
        progress_bar.update(1)
    
    progress_bar.close()
    
    # Final results
    print("\n==== Grid Search Complete ====")
    print(f"Best configuration:")
    print(json.dumps(best_config, indent=2))
    print(f"Best BLEU score: {best_bleu:.5f}")
    
    return best_config, best_bleu

def apply_best_config_to_test(df_train_prompts, df_train_responses, df_dev_prompts, df_dev_responses, 
                            df_test_prompts, all_texts, best_config, model_dir, output_dir):
    """
    Apply the best configuration to generate test predictions
    """
    print("\nApplying best configuration to test set...")
    
    # Create combined candidate pool for test prediction
    candidate_prompts = pd.concat([df_train_prompts, df_dev_prompts], ignore_index=True)
    candidate_responses = pd.concat([df_train_responses, df_dev_responses], ignore_index=True)
    
    # Train or load the best model
    model = train_model(all_texts, best_config, model_dir)
    
    # Get embedding method and other config parameters
    embedding_method = best_config["embedding_method"]
    remove_punct = best_config.get("remove_punct", True)
    
    # Calculate word weights if using weighted embedding
    word_weights = None
    if embedding_method == "weighted":
        word_weights = calculate_word_weights(all_texts, remove_punct)
    
    # Generate embeddings
    print("Generating embeddings for candidate prompts...")
    candidate_embeddings = np.array([
        get_sentence_embedding(model, text, embedding_method, word_weights, remove_punct) 
        for text in tqdm(candidate_prompts['user_prompt'])
    ])
    
    print("Generating embeddings for test prompts...")
    test_embeddings = np.array([
        get_sentence_embedding(model, text, embedding_method, word_weights, remove_punct) 
        for text in tqdm(df_test_prompts['user_prompt'])
    ])
    
    # Compute similarity and find best matches
    print("Computing similarity matrix...")
    similarity_matrix = cosine_similarity(test_embeddings, candidate_embeddings)
    best_indices = np.argmax(similarity_matrix, axis=1)
    best_candidate_ids = candidate_prompts.iloc[best_indices]['conversation_id'].values
    
    # Create test submission file
    test_results_df = pd.DataFrame({
        "conversation_id": df_test_prompts['conversation_id'],
        "response_id": best_candidate_ids
    })
    
    # Save test predictions
    test_output_csv = os.path.join(output_dir, "track_2_test.csv")
    test_results_df.to_csv(test_output_csv, index=False)
    print(f"Test predictions saved to: {test_output_csv}")
    
    # Save best config info
    config_info = {
        "best_config": best_config,
        "description": "This configuration was found by grid search to have the highest BLEU score on the dev set."
    }
    
    config_output_file = os.path.join(output_dir, "track_2_best_config.json")
    with open(config_output_file, 'w') as f:
        json.dump(config_info, f, indent=2)
    
    print(f"Configuration details saved to: {config_output_file}")

def main():
    parser = argparse.ArgumentParser(description="Track 2: Grid Search for Distributed Representations")
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
    
    results_file = os.path.join(results_dir, "track_2_grid_search_results.json")
    
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
    
    # Combine all texts for training
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
            best_config, best_bleu = run_grid_search(
                df_train_prompts, df_train_responses,
                df_dev_prompts, df_dev_responses,
                results_file, model_dir
            )
    else:
        # Run grid search
        best_config, best_bleu = run_grid_search(
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