#!/usr/bin/env python3
import os
import argparse
import json
import numpy as np
import pandas as pd
import re
import string
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import multiprocessing
from gensim.models import FastText, Word2Vec, KeyedVectors
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess

# ===================== PARAMETERS =====================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "..", "data")
DUMP_DIR = os.path.join(CURRENT_DIR, "..", "dump")
RESULTS_DIR = os.path.join(CURRENT_DIR, "..", "results")
MODEL_DIR = os.path.join(CURRENT_DIR, "..", "models")
os.makedirs(DUMP_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Response filtering
MIN_RESPONSE_LENGTH = 3  # Min number of characters for valid responses

# Grid search parameters
EMBEDDING_METHODS = [
    "fasttext",      # FastText with subword information
    "word2vec",      # Word2Vec skip-gram
    "doc2vec",       # Paragraph embeddings
]

VECTOR_SIZES = [100, 200, 300]  # Embedding dimensions
WINDOW_SIZES = [3, 5, 10]  # Context window sizes
MIN_COUNTS = [2, 5]  # Minimum word counts
EPOCHS = [5, 10, 20]  # Training epochs

# Similarity metrics
SIMILARITY_METRICS = [
    "cosine",
    "euclidean",
    "manhattan"
]

# Aggregation methods for word vectors to sentence vectors
AGGREGATION_METHODS = [
    "mean",            # Simple averaging
    "tfidf_weighted",  # TF-IDF weighted averaging (not for Doc2Vec)
]

# Results tracking
GRID_SEARCH_RESULTS_FILE = os.path.join(RESULTS_DIR, "track2_grid_search_results.json")
# ======================================================

def preprocess_text(text):
    """
    Basic preprocessing:
    - Convert to string
    - Lowercase
    - Normalize whitespace
    """
    if pd.isna(text):
        return ""
    text = str(text).lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def simple_tokenize(text):
    """
    Simple tokenizer that doesn't rely on NLTK.
    Just splits on whitespace and removes punctuation.
    """
    if not text or pd.isna(text):
        return []
    
    # Replace punctuation with spaces
    for punct in string.punctuation:
        text = text.replace(punct, ' ')
    
    # Split on whitespace and filter out empty tokens
    tokens = [token for token in text.split() if token]
    return tokens

def tag_documents(texts, ids):
    """
    Create tagged documents for Doc2Vec training
    """
    return [TaggedDocument(simple_tokenize(text), [str(id)]) for text, id in zip(texts, ids)]

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

def train_embedding_model(tokenized_texts, config, train_ids=None):
    """
    Train an embedding model based on the specified configuration
    """
    method = config['embedding_method']
    vector_size = config['vector_size']
    window_size = config['window_size']
    min_count = config['min_count']
    epochs = config['epochs']
    workers = max(1, multiprocessing.cpu_count() - 1)
    
    model_name = f"{method}_vs{vector_size}_ws{window_size}_mc{min_count}_e{epochs}"
    model_path = os.path.join(MODEL_DIR, model_name)
    
    # Check if model already exists
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        if method == "doc2vec":
            return Doc2Vec.load(model_path)
        elif method == "fasttext":
            return FastText.load(model_path)
        else:  # word2vec
            return Word2Vec.load(model_path)
    
    print(f"Training {method} model with vector_size={vector_size}, window={window_size}, min_count={min_count}, epochs={epochs}")
    
    if method == "doc2vec":
        # Doc2Vec needs tagged documents
        if train_ids is None:
            train_ids = list(range(len(tokenized_texts)))
        
        tagged_docs = [TaggedDocument(tokens, [str(id)]) for tokens, id in zip(tokenized_texts, train_ids)]
        
        model = Doc2Vec(
            tagged_docs,
            vector_size=vector_size,
            window=window_size,
            min_count=min_count,
            epochs=epochs,
            workers=workers
        )
    elif method == "fasttext":
        model = FastText(
            tokenized_texts,
            vector_size=vector_size,
            window=window_size,
            min_count=min_count,
            epochs=epochs,
            workers=workers
        )
    else:  # word2vec
        model = Word2Vec(
            tokenized_texts,
            vector_size=vector_size,
            window=window_size,
            min_count=min_count,
            epochs=epochs,
            workers=workers
        )
    
    # Save model for reuse
    model.save(model_path)
    return model

def compute_tfidf_weights(tokenized_texts):
    """
    Compute TF-IDF weights for words in the corpus
    """
    # Count documents where each word appears
    word_doc_counts = {}
    for doc in tokenized_texts:
        # Count each word only once per document
        for word in set(doc):
            word_doc_counts[word] = word_doc_counts.get(word, 0) + 1
    
    # Total document count
    num_docs = len(tokenized_texts)
    
    # Compute IDF values
    idf = {}
    for word, doc_count in word_doc_counts.items():
        idf[word] = np.log(num_docs / (1 + doc_count))
    
    return idf

def get_sentence_embedding(model, sentence_tokens, config, idf=None):
    """
    Calculate sentence embedding based on configuration
    """
    if not sentence_tokens:
        # Return zero vector for empty sentences
        if config['embedding_method'] == "doc2vec":
            # For Doc2Vec, infer vector directly
            return np.zeros(config['vector_size'])
        else:
            return np.zeros(model.vector_size)
    
    if config['embedding_method'] == "doc2vec":
        # For Doc2Vec, use the model directly
        return model.infer_vector(sentence_tokens)
    
    # For Word2Vec and FastText, we need to aggregate word vectors
    word_vectors = []
    word_weights = []
    
    for word in sentence_tokens:
        try:
            if config['embedding_method'] == "fasttext":
                # FastText can handle OOV words
                word_vectors.append(model.wv[word])
            else:  # word2vec
                if word in model.wv:
                    word_vectors.append(model.wv[word])
        except KeyError:
            # Skip words not in vocabulary
            continue
    
    if not word_vectors:
        return np.zeros(model.vector_size)
    
    # Apply aggregation method
    if config['aggregation_method'] == "tfidf_weighted" and idf:
        # TF-IDF weighted averaging
        for i, word in enumerate(sentence_tokens):
            if word in idf:
                word_weights.append(idf[word])
            else:
                word_weights.append(1.0)
        
        # Normalize weights
        total_weight = sum(word_weights)
        if total_weight > 0:
            word_weights = [w / total_weight for w in word_weights]
        
        # Weighted average
        sentence_vector = np.average(word_vectors, axis=0, weights=word_weights[:len(word_vectors)])
    else:
        # Simple averaging
        sentence_vector = np.mean(word_vectors, axis=0)
    
    # Normalize to unit length
    norm = np.linalg.norm(sentence_vector)
    if norm > 0:
        sentence_vector = sentence_vector / norm
    
    return sentence_vector

def calculate_similarity(embeddings1, embeddings2, metric):
    """
    Calculate similarity matrix using different metrics
    """
    if metric == "cosine":
        return cosine_similarity(embeddings1, embeddings2)
    elif metric == "euclidean":
        # Convert distance to similarity (higher is better)
        distances = euclidean_distances(embeddings1, embeddings2)
        # Normalize to [0, 1] range and invert
        max_dist = np.max(distances)
        if max_dist > 0:
            return 1 - (distances / max_dist)
        return np.zeros_like(distances)
    elif metric == "manhattan":
        # Convert distance to similarity (higher is better)
        distances = manhattan_distances(embeddings1, embeddings2)
        # Normalize to [0, 1] range and invert
        max_dist = np.max(distances)
        if max_dist > 0:
            return 1 - (distances / max_dist)
        return np.zeros_like(distances)
    else:
        # Default to cosine similarity
        return cosine_similarity(embeddings1, embeddings2)

def evaluate_configuration(df_train_prompts, df_train_responses, df_dev_prompts, df_dev_responses, config):
    """
    Evaluate a specific configuration and return the BLEU score
    """
    # Process and tokenize text
    df_train_prompts['processed'] = df_train_prompts['user_prompt'].apply(preprocess_text)
    df_dev_prompts['processed'] = df_dev_prompts['user_prompt'].apply(preprocess_text)
    
    train_tokenized = [simple_tokenize(text) for text in df_train_prompts['processed']]
    dev_tokenized = [simple_tokenize(text) for text in df_dev_prompts['processed']]
    
    # Train embedding model
    model = train_embedding_model(
        train_tokenized, 
        config, 
        train_ids=df_train_prompts['conversation_id'].tolist()
    )
    
    # Compute TF-IDF weights if needed
    idf = None
    if config['aggregation_method'] == "tfidf_weighted" and config['embedding_method'] != "doc2vec":
        idf = compute_tfidf_weights(train_tokenized)
    
    # Generate embeddings for all prompts
    print("Generating embeddings for training prompts...")
    train_embeddings = np.array([
        get_sentence_embedding(model, tokens, config, idf) 
        for tokens in tqdm(train_tokenized, desc="Train embeddings")
    ])
    
    print("Generating embeddings for dev prompts...")
    dev_embeddings = np.array([
        get_sentence_embedding(model, tokens, config, idf) 
        for tokens in tqdm(dev_tokenized, desc="Dev embeddings")
    ])
    
    # Calculate similarities
    similarity_matrix = calculate_similarity(dev_embeddings, train_embeddings, config['similarity_metric'])
    
    # Exclude self-matches for dev evaluation
    dev_ids = df_dev_prompts['conversation_id'].tolist()
    train_ids = df_train_prompts['conversation_id'].tolist()
    
    for i, dev_id in enumerate(dev_ids):
        for j, train_id in enumerate(train_ids):
            if dev_id == train_id:
                similarity_matrix[i, j] = -float('inf')  # Force non-selection
    
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
    
    # Calculate total configurations
    total_configs = 0
    for method in EMBEDDING_METHODS:
        for vs in VECTOR_SIZES:
            for ws in WINDOW_SIZES:
                for mc in MIN_COUNTS:
                    for ep in EPOCHS:
                        for sim in SIMILARITY_METRICS:
                            # Doc2Vec doesn't use aggregation methods
                            if method == "doc2vec":
                                total_configs += 1
                            else:
                                total_configs += len(AGGREGATION_METHODS)
    
    print(f"Starting grid search with {total_configs} configurations...")
    config_count = 0
    
    # Create progress bar
    progress_bar = tqdm(total=total_configs, desc="Evaluating configurations")
    
    # Grid search over all parameters
    for method in EMBEDDING_METHODS:
        for vector_size in VECTOR_SIZES:
            for window_size in WINDOW_SIZES:
                for min_count in MIN_COUNTS:
                    for epochs in EPOCHS:
                        for similarity_metric in SIMILARITY_METRICS:
                            # Set aggregation methods based on embedding type
                            if method == "doc2vec":
                                aggregation_methods = ["none"]  # Doc2Vec doesn't need aggregation
                            else:
                                aggregation_methods = AGGREGATION_METHODS
                            
                            for aggregation_method in aggregation_methods:
                                config = {
                                    'embedding_method': method,
                                    'vector_size': vector_size,
                                    'window_size': window_size,
                                    'min_count': min_count,
                                    'epochs': epochs,
                                    'similarity_metric': similarity_metric,
                                    'aggregation_method': aggregation_method
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
    
    # Combine train and dev for the candidate pool
    candidate_prompts = pd.concat([df_train_prompts, df_dev_prompts], ignore_index=True)
    
    # Process and tokenize text
    candidate_prompts['processed'] = candidate_prompts['user_prompt'].apply(preprocess_text)
    df_test_prompts['processed'] = df_test_prompts['user_prompt'].apply(preprocess_text)
    
    candidate_tokenized = [simple_tokenize(text) for text in candidate_prompts['processed']]
    test_tokenized = [simple_tokenize(text) for text in df_test_prompts['processed']]
    
    # Train embedding model on all available data
    model = train_embedding_model(
        candidate_tokenized, 
        best_config, 
        train_ids=candidate_prompts['conversation_id'].tolist()
    )
    
    # Compute TF-IDF weights if needed
    idf = None
    if best_config['aggregation_method'] == "tfidf_weighted" and best_config['embedding_method'] != "doc2vec":
        idf = compute_tfidf_weights(candidate_tokenized)
    
    # Generate embeddings for all prompts
    print("Generating embeddings for candidate prompts...")
    candidate_embeddings = np.array([
        get_sentence_embedding(model, tokens, best_config, idf) 
        for tokens in tqdm(candidate_tokenized, desc="Candidate embeddings")
    ])
    
    print("Generating embeddings for test prompts...")
    test_embeddings = np.array([
        get_sentence_embedding(model, tokens, best_config, idf) 
        for tokens in tqdm(test_tokenized, desc="Test embeddings")
    ])
    
    # Calculate similarities
    similarity_matrix = calculate_similarity(test_embeddings, candidate_embeddings, best_config['similarity_metric'])
    
    # Find best matches
    best_indices = np.argmax(similarity_matrix, axis=1)
    best_candidate_ids = candidate_prompts.iloc[best_indices]['conversation_id'].values
    
    # Create test results
    test_results_df = pd.DataFrame({
        "conversation_id": df_test_prompts['conversation_id'],
        "response_id": best_candidate_ids
    })
    
    # Save results
    test_output_csv = os.path.join(DUMP_DIR, "track_2_best_config_test.csv")
    test_results_df.to_csv(test_output_csv, index=False)
    print(f"Test predictions saved to: {test_output_csv}")
    
    # Save config information for reproducibility
    config_info = {
        "best_config": best_config,
        "description": "This configuration was found by grid search to have the highest BLEU score on the dev set."
    }
    
    config_output_file = os.path.join(RESULTS_DIR, "track_2_best_config.json")
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