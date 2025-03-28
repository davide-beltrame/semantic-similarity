import pandas as pd
import numpy as np
import re
import os
import string
import argparse
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm
from gensim.models import Word2Vec
import multiprocessing

# Response filtering
MIN_RESPONSE_LENGTH = 3  # Min number of characters for valid responses

def preprocess_text(text, remove_punct=True):
    """
    Preprocess text for Word2Vec:
    - Convert to string
    - Lowercase
    - Normalize whitespace
    - Optionally remove punctuation
    """
    if pd.isna(text):
        return ""
        
    text = str(text).lower().strip()
    text = re.sub(r'\s+', ' ', text)
    
    if remove_punct:
        # Remove punctuation and replace with spaces
        for punct in string.punctuation:
            text = text.replace(punct, ' ')
        # Normalize whitespace again
        text = re.sub(r'\s+', ' ', text)
        
    return text

def simple_tokenize(text, remove_punct=True):
    """
    Simple tokenizer that splits on whitespace and optionally removes punctuation.
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

def train_word2vec_model(texts, config, model_path):
    """
    Train a Word2Vec model on the given texts using the specified configuration.
    """
    print("Tokenizing texts for Word2Vec training...")
    
    vector_size = config["vector_size"]
    window = config["window"]
    min_count = config["min_count"]
    epochs = config["epochs"]
    remove_punct = config["remove_punct"]
    
    tokenized_texts = [simple_tokenize(text, remove_punct) for text in tqdm(texts)]
    
    # Remove empty lists
    tokenized_texts = [tokens for tokens in tokenized_texts if tokens]
    
    print(f"Training Word2Vec model on {len(tokenized_texts)} documents...")
    # Use as many workers as CPU cores
    workers = max(1, multiprocessing.cpu_count() - 1)
    
    # Train Word2Vec model with best configuration
    model = Word2Vec(
        tokenized_texts,
        vector_size=vector_size,  
        window=window,
        min_count=min_count,
        epochs=epochs,
        workers=workers
    )
    
    print(f"Saving model to {model_path}")
    model.save(model_path)
    
    return model

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

def calculate_word_frequency(tokenized_texts):
    """
    Calculate word frequency for weighting.
    Uses inverse document frequency approach to give higher weights to rare words.
    """
    # Count documents (sentences) where each word appears
    doc_counts = {}
    total_docs = len(tokenized_texts)
    
    for tokens in tokenized_texts:
        # Count each word only once per document
        unique_tokens = set(tokens)
        for token in unique_tokens:
            doc_counts[token] = doc_counts.get(token, 0) + 1
    
    # Compute weights as inverse document frequency
    # Formula: log(N/df) where N is total documents and df is document frequency
    word_weights = {}
    for word, count in doc_counts.items():
        # Add small constant (1) to avoid division by zero and log(1) = 0
        idf = np.log(total_docs / (count + 1)) + 1.0
        word_weights[word] = idf
    
    return word_weights

def get_sentence_embedding(model, sentence, word_weights, method="weighted", remove_punct=True):
    """
    Calculate sentence embedding using different methods:
    - mean: Simple average of word vectors
    - weighted: Weighted average based on inverse document frequency
    - max_pooling: Take the maximum value for each dimension
    """
    if not sentence or sentence.isspace():
        # Return zero vector for empty sentences
        return np.zeros(model.vector_size)
    
    # Tokenize the sentence
    words = simple_tokenize(sentence, remove_punct)
    
    if not words:
        return np.zeros(model.vector_size)
    
    # Get embeddings for all words
    word_vectors = []
    weights = []
    
    for word in words:
        try:
            # Get word vector
            word_vector = model.wv[word]
            
            # Get weight from word_weights dictionary
            weight = word_weights.get(word, 1.0)  # Default to 1.0 if not in vocabulary
            
            word_vectors.append(word_vector)
            weights.append(weight)
        except KeyError:
            # Skip words not in vocabulary
            continue
    
    if not word_vectors:
        return np.zeros(model.vector_size)
    
    # Convert to numpy arrays
    word_vectors = np.array(word_vectors)
    
    # Calculate embedding based on specified method
    if method == "mean":
        # Simple average
        sentence_vector = np.mean(word_vectors, axis=0)
    
    elif method == "weighted":
        # Weighted average using IDF weights
        weights = np.array(weights).reshape(-1, 1)  # Column vector for broadcasting
        sentence_vector = np.sum(word_vectors * weights, axis=0) / np.sum(weights)
    
    elif method == "max_pooling":
        # Max pooling across each dimension
        sentence_vector = np.max(word_vectors, axis=0)
    
    else:
        # Default to weighted if method is unknown
        weights = np.array(weights).reshape(-1, 1)
        sentence_vector = np.sum(word_vectors * weights, axis=0) / np.sum(weights)
    
    # Normalize to unit length
    norm = np.linalg.norm(sentence_vector)
    if norm > 0:
        sentence_vector = sentence_vector / norm
    
    return sentence_vector

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Track 2: Word2Vec with Weighted Embeddings")
    parser.add_argument("--no_filter", action="store_true", help="Don't filter invalid responses")
    parser.add_argument("--output_dir", type=str, default=None, 
                        help="Directory for output files (default: parent directory of script)")
    args = parser.parse_args()
    
    filter_responses = not args.no_filter
    
    # Set up paths relative to this script's location
    code_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(code_dir)  # Parent directory
    data_dir = os.path.join(parent_dir, "data")
    model_dir = os.path.join(parent_dir, "models")
    
    # Set output directory to parent directory by default
    output_dir = args.output_dir if args.output_dir else parent_dir
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # Best configuration from grid search
    best_config = {
        "model_type": "word2vec",
        "vector_size": 300,
        "window": 5,
        "min_count": 2,
        "epochs": 20,
        "embedding_method": "weighted",
        "remove_punct": True
    }
    
    # Create model filename based on configuration
    model_filename = f"word2vec_vs{best_config['vector_size']}_win{best_config['window']}_mc{best_config['min_count']}_ep{best_config['epochs']}.bin"
    model_path = os.path.join(model_dir, model_filename)
    
    # Load all data
    print("Loading data...")
    df_dev_prompts = pd.read_csv(os.path.join(data_dir, "dev_prompts.csv"))
    df_dev_responses = pd.read_csv(os.path.join(data_dir, "dev_responses.csv"))
    df_train_prompts = pd.read_csv(os.path.join(data_dir, "train_prompts.csv"))
    df_train_responses = pd.read_csv(os.path.join(data_dir, "train_responses.csv"))
    df_test_prompts = pd.read_csv(os.path.join(data_dir, "test_prompts.csv"))
    
    # Apply response filtering if requested
    if filter_responses:
        df_train_prompts, df_train_responses = filter_invalid_responses(df_train_prompts, df_train_responses)
    
    # Preprocess all text based on configuration
    remove_punct = best_config["remove_punct"]
    print(f"Preprocessing texts (remove_punct={remove_punct})...")
    
    # Combine all texts for training
    all_texts = pd.concat([
        df_train_prompts['user_prompt'],
        df_dev_prompts['user_prompt'],
        df_test_prompts['user_prompt']
    ]).tolist()
    
    # Add responses to the training corpus to capture response patterns
    all_responses = pd.concat([
        df_train_responses['model_response'],
        df_dev_responses['model_response']
    ]).dropna().tolist()
    
    all_texts.extend(all_responses)
    
    # Train or load Word2Vec model
    if os.path.exists(model_path):
        print(f"Loading existing Word2Vec model from {model_path}")
        word2vec_model = Word2Vec.load(model_path)
    else:
        print("Training new Word2Vec model...")
        word2vec_model = train_word2vec_model(all_texts, best_config, model_path)
    
    # Tokenize all texts for frequency calculation
    print("Tokenizing texts for frequency calculation...")
    tokenized_texts = [simple_tokenize(text, remove_punct) for text in all_texts]
    
    # Calculate word weights based on inverse document frequency
    print("Calculating word weights based on inverse document frequency...")
    word_weights = calculate_word_frequency(tokenized_texts)
    
    print(f"Using embedding method: {best_config['embedding_method']}")
    
    # Generate weighted embeddings for training prompts
    print("Generating weighted embeddings for training prompts...")
    train_embeddings = np.array([
        get_sentence_embedding(word2vec_model, text, word_weights, remove_punct) 
        for text in tqdm(df_train_prompts['user_prompt'])
    ])
    
    # Generate weighted embeddings for dev prompts
    print("Generating weighted embeddings for dev prompts...")
    dev_embeddings = np.array([
        get_sentence_embedding(word2vec_model, text, word_weights, remove_punct) 
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
    
    # Compute BLEU scores for dev evaluation
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
    
    # Create suffix for output files based on filtering
    suffix = "_filtered" if filter_responses else ""
    
    # Save dev evaluation results
    results_df = pd.DataFrame({
        "dev_conversation_id": df_dev_prompts['conversation_id'],
        "retrieved_train_id": best_train_ids,
        "true_response": df_dev_prompts['conversation_id'].map(dev_response_map),
        "predicted_response": [train_response_map.get(rid, "") for rid in best_train_ids],
        "bleu_score": bleu_scores
    })
    
    dev_output_csv = os.path.join(output_dir, f"track_2_dev{suffix}.csv")
    results_df.to_csv(dev_output_csv, index=False)
    
    # Print the average BLEU score
    print(f"Average BLEU score on dev set: {avg_bleu:.5f}")
    
    # Generate test predictions
    # For test set, use combined train+dev as the candidate pool
    candidate_prompts = pd.concat([df_train_prompts, df_dev_prompts], ignore_index=True)
    
    # Generate embeddings for candidates and test prompts
    print("Generating weighted embeddings for candidate prompts...")
    candidate_embeddings = np.array([
        get_sentence_embedding(word2vec_model, text, word_weights, remove_punct) 
        for text in tqdm(candidate_prompts['user_prompt'])
    ])
    
    print("Generating weighted embeddings for test prompts...")
    test_embeddings = np.array([
        get_sentence_embedding(word2vec_model, text, word_weights, remove_punct) 
        for text in tqdm(df_test_prompts['user_prompt'])
    ])
    
    # Compute similarities and find best matches
    print("Computing test similarities...")
    test_similarity_matrix = cosine_similarity(test_embeddings, candidate_embeddings)
    test_best_indices = np.argmax(test_similarity_matrix, axis=1)
    test_best_candidate_ids = candidate_prompts.iloc[test_best_indices]['conversation_id'].values
    
    # Create and save test submission file
    test_results_df = pd.DataFrame({
        "conversation_id": df_test_prompts['conversation_id'],
        "response_id": test_best_candidate_ids
    })
    
    # Save as track_2_test.csv in the output directory
    test_output_csv = os.path.join(output_dir, "track_2_test.csv")
    test_results_df.to_csv(test_output_csv, index=False)
    
    print(f"Dev evaluation saved to: {dev_output_csv}")
    print(f"Test predictions saved to: {test_output_csv}")

if __name__ == "__main__":
    main()