import pandas as pd
import numpy as np
import re
import os
import string
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm
from gensim.models import FastText
import multiprocessing
from sklearn.feature_extraction.text import TfidfVectorizer

# Response filtering
MIN_RESPONSE_LENGTH = 3  # Min number of characters for valid responses

def preprocess_text(text):
    """
    Preprocess text for FastText:
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
    
    # Remove punctuation and replace with spaces
    for punct in string.punctuation:
        text = text.replace(punct, ' ')
    
    # Split on whitespace and filter out empty tokens
    tokens = [token for token in text.split() if token]
    return tokens

def train_fasttext_model(texts, model_path="fasttext_model.bin"):
    """
    Train a FastText model on the given texts and save it.
    """
    print("Tokenizing texts for FastText training...")
    tokenized_texts = [simple_tokenize(text) for text in tqdm(texts)]
    
    # Remove empty lists
    tokenized_texts = [tokens for tokens in tokenized_texts if tokens]
    
    print(f"Training FastText model on {len(tokenized_texts)} documents...")
    # Use as many workers as CPU cores
    workers = max(1, multiprocessing.cpu_count() - 1)
    
    # Train FastText model with best configuration and speed optimizations
    model = FastText(
        tokenized_texts,
        vector_size=300,    # Best config: vector_size=300
        window=3,           # Best config: window_size=3
        min_count=5,        # Best config: min_count=5
        epochs=20,          # Best config: epochs=20
        workers=workers,    # Parallel training threads
        sg=1,               # Use skip-gram (faster than CBOW for training)
        negative=5,         # Reduce negative samples (default is 5-10)
        sample=1e-4,        # Downsampling of frequent words
        min_n=3,            # Minimum length of char ngrams
        max_n=6,            # Maximum length of char ngrams
        bucket=2000000,     # Hash table buckets for character ngrams
        callbacks=[]        # Disable callbacks to speed up training
    )
    
    print(f"Saving model to {model_path}")
    model.save(model_path)
    
    return model

def get_tfidf_weights(texts):
    """
    Calculate TF-IDF weights for words in the corpus
    """
    print("Calculating TF-IDF weights...")
    # Create TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(
        tokenizer=simple_tokenize,
        lowercase=True,
        use_idf=True,
        norm='l2',
        smooth_idf=True
    )
    
    # Fit vectorizer to get IDF values
    tfidf_vectorizer.fit(texts)
    
    # Get the vocabulary and idf values
    vocabulary = tfidf_vectorizer.vocabulary_
    idf_values = tfidf_vectorizer.idf_
    
    # Create a word -> tfidf weight dictionary
    word_tfidf = {word: idf_values[vocabulary[word]] for word in vocabulary}
    
    return word_tfidf, tfidf_vectorizer

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

def get_sentence_embedding_tfidf_weighted(model, sentence, word_tfidf):
    """
    Calculate sentence embedding by weighting FastText word vectors with TF-IDF.
    This implements the 'tfidf_weighted' aggregation method from the best configuration.
    """
    if not sentence or sentence.isspace():
        # Return zero vector for empty sentences
        return np.zeros(model.vector_size)
    
    # Tokenize the sentence
    words = simple_tokenize(sentence)
    
    if not words:
        return np.zeros(model.vector_size)
    
    # Count word frequencies in this sentence (term frequency)
    word_counts = {}
    for word in words:
        word_counts[word] = word_counts.get(word, 0) + 1
    
    # Get embeddings for all words weighted by TF-IDF
    word_vectors = []
    weights = []
    
    for word in words:
        try:
            # Get word vector
            word_vector = model.wv[word]
            
            # Calculate weight: term frequency * inverse document frequency
            tf = word_counts[word] / len(words)  # Normalized term frequency
            idf = word_tfidf.get(word, 1.0)  # Use default 1.0 if word not in TF-IDF vocab
            weight = tf * idf
            
            word_vectors.append(word_vector)
            weights.append(weight)
        except KeyError:
            # Skip words not in vocabulary (shouldn't happen much with FastText)
            continue
    
    if not word_vectors:
        return np.zeros(model.vector_size)
    
    # Convert to numpy arrays
    word_vectors = np.array(word_vectors)
    weights = np.array(weights).reshape(-1, 1)  # Column vector for broadcasting
    
    # Weighted average of word vectors
    sentence_vector = np.sum(word_vectors * weights, axis=0) / np.sum(weights)
    
    # Normalize to unit length
    norm = np.linalg.norm(sentence_vector)
    if norm > 0:
        sentence_vector = sentence_vector / norm
    
    return sentence_vector

def main():
    # Set up paths relative to this script's location
    code_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(code_dir, "../data")
    dump_dir = os.path.join(code_dir, "../dump")
    model_dir = os.path.join(code_dir, "../models")
    os.makedirs(dump_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    fasttext_model_path = os.path.join(model_dir, "fasttext_model.bin")
    
    # Load all data
    print("Loading data...")
    df_dev_prompts = pd.read_csv(os.path.join(data_dir, "dev_prompts.csv"))
    df_dev_responses = pd.read_csv(os.path.join(data_dir, "dev_responses.csv"))
    df_train_prompts = pd.read_csv(os.path.join(data_dir, "train_prompts.csv"))
    df_train_responses = pd.read_csv(os.path.join(data_dir, "train_responses.csv"))
    df_test_prompts = pd.read_csv(os.path.join(data_dir, "test_prompts.csv"))
    
    # Apply response filtering
    print("Applying response filtering...")
    df_train_prompts, df_train_responses = filter_invalid_responses(df_train_prompts, df_train_responses)
    
    # Preprocess all prompts
    print("Preprocessing prompts...")
    df_train_prompts['processed'] = df_train_prompts['user_prompt'].apply(preprocess_text)
    df_dev_prompts['processed'] = df_dev_prompts['user_prompt'].apply(preprocess_text)
    df_test_prompts['processed'] = df_test_prompts['user_prompt'].apply(preprocess_text)
    
    # For faster training, use a subset of data for model training
    # Combine all texts in a more efficient way
    print("Preparing training corpus...")
    all_prompts = pd.concat([
        df_train_prompts['processed'],
        df_dev_prompts['processed'], 
        df_test_prompts['processed']
    ]).dropna().tolist()
    
    # Only include a subset of responses to speed up training
    sample_size = min(10000, len(df_train_responses))  # Cap at 10k responses
    sampled_responses = pd.concat([
        df_train_responses['model_response'].sample(sample_size, random_state=42),
        df_dev_responses['model_response'].sample(min(1000, len(df_dev_responses)), random_state=42)
    ]).dropna().apply(preprocess_text).tolist()
    
    all_texts = all_prompts + sampled_responses
    
    # Check if model exists, train if not
    if os.path.exists(fasttext_model_path):
        print(f"Loading existing FastText model from {fasttext_model_path}")
        fasttext_model = FastText.load(fasttext_model_path)
    else:
        print("Training new FastText model...")
        fasttext_model = train_fasttext_model(all_texts, fasttext_model_path)
    
    # Calculate TF-IDF weights more efficiently by using a lower max_features
    print("Calculating TF-IDF weights (optimized)...")
    tfidf_vectorizer = TfidfVectorizer(
        tokenizer=simple_tokenize,
        lowercase=True,
        use_idf=True,
        norm='l2',
        smooth_idf=True,
        max_features=50000  # Limit vocabulary size for speed
    )
    
    # Fit vectorizer to get IDF values
    tfidf_vectorizer.fit(all_texts)
    
    # Get the vocabulary and idf values
    vocabulary = tfidf_vectorizer.vocabulary_
    idf_values = tfidf_vectorizer.idf_
    
    # Create a word -> tfidf weight dictionary
    word_tfidf = {word: idf_values[vocabulary[word]] for word in vocabulary}
    
    # Generate embeddings more efficiently using batching
    print("Generating TF-IDF weighted embeddings for training prompts...")
    
    # Process in smaller batches to reduce memory usage
    batch_size = 1000
    train_embeddings = []
    
    for i in tqdm(range(0, len(df_train_prompts), batch_size)):
        batch = df_train_prompts['processed'][i:i+batch_size]
        batch_embeddings = np.array([
            get_sentence_embedding_tfidf_weighted(fasttext_model, text, word_tfidf) 
            for text in batch
        ])
        train_embeddings.append(batch_embeddings)
    
    # Combine batches
    train_embeddings = np.vstack(train_embeddings)
    
    print("Generating TF-IDF weighted embeddings for dev prompts...")
    dev_embeddings = np.array([
        get_sentence_embedding_tfidf_weighted(fasttext_model, text, word_tfidf) 
        for text in tqdm(df_dev_prompts['processed'])
    ])
    
    # Compute similarity and find best matches (using cosine similarity as specified)
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
    
    # Save dev evaluation results
    results_df = pd.DataFrame({
        "dev_conversation_id": df_dev_prompts['conversation_id'],
        "retrieved_train_id": best_train_ids,
        "true_response": df_dev_prompts['conversation_id'].map(dev_response_map),
        "predicted_response": [train_response_map.get(rid, "") for rid in best_train_ids],
        "bleu_score": bleu_scores
    })
    output_csv = os.path.join(dump_dir, "dev_bleu_evaluation_fasttext.csv")
    results_df.to_csv(output_csv, index=False)
    
    # Process test set in batches
    print("Generating TF-IDF weighted embeddings for test prompts...")
    test_embeddings = []
    
    for i in tqdm(range(0, len(df_test_prompts), batch_size)):
        batch = df_test_prompts['processed'][i:i+batch_size]
        batch_embeddings = np.array([
            get_sentence_embedding_tfidf_weighted(fasttext_model, text, word_tfidf) 
            for text in batch
        ])
        test_embeddings.append(batch_embeddings)
    
    # Combine batches
    test_embeddings = np.vstack(test_embeddings)
    
    print("Computing test similarities...")
    test_similarity_matrix = cosine_similarity(test_embeddings, train_embeddings)
    test_best_indices = np.argmax(test_similarity_matrix, axis=1)
    test_best_train_ids = df_train_prompts.iloc[test_best_indices]['conversation_id'].values
    
    # Create test submission file
    test_results_df = pd.DataFrame({
        "conversation_id": df_test_prompts['conversation_id'],
        "response_id": test_best_train_ids
    })
    test_output_csv = os.path.join(dump_dir, "track_2_test.csv")
    test_results_df.to_csv(test_output_csv, index=False)
    
    print(f"Average BLEU score on dev set: {avg_bleu:.5f}")
    print(f"Dev evaluation saved to: {output_csv}")
    print(f"Test predictions saved to: {test_output_csv}")

if __name__ == "__main__":
    main()