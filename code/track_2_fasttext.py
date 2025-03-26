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
    
    # Train FastText model - adjust parameters as needed
    model = FastText(
        tokenized_texts,
        vector_size=300,  # Dimension of the word vectors
        window=5,         # Context window size
        #min_count=2,      # Minimum word count threshold
        epochs=100,        # Number of training epochs
        workers=workers   # Parallel training threads
    )
    
    print(f"Saving model to {model_path}")
    model.save(model_path)
    
    return model

def get_sentence_embedding(model, sentence):
    """
    Calculate sentence embedding by averaging FastText word vectors.
    """
    if not sentence or sentence.isspace():
        # Return zero vector for empty sentences
        return np.zeros(model.vector_size)
    
    # Tokenize the sentence
    words = simple_tokenize(sentence)
    
    if not words:
        return np.zeros(model.vector_size)
    
    # Get embeddings for all words (FastText can handle OOV words)
    word_vectors = []
    for word in words:
        try:
            word_vectors.append(model.wv[word])
        except KeyError:
            # Skip words not in vocabulary (shouldn't happen much with FastText)
            continue
    
    if not word_vectors:
        return np.zeros(model.vector_size)
    
    # Average the word vectors
    sentence_vector = np.mean(word_vectors, axis=0)
    
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
    
    # Preprocess all prompts
    print("Preprocessing prompts...")
    df_train_prompts['processed'] = df_train_prompts['user_prompt'].apply(preprocess_text)
    df_dev_prompts['processed'] = df_dev_prompts['user_prompt'].apply(preprocess_text)
    df_test_prompts['processed'] = df_test_prompts['user_prompt'].apply(preprocess_text)
    
    # Combine all texts for training (train + dev + test prompts)
    all_texts = pd.concat([
        df_train_prompts['processed'],
        df_dev_prompts['processed'], 
        df_test_prompts['processed']
    ]).tolist()
    
    # Add responses to the training corpus to capture response patterns too
    all_responses = pd.concat([
        df_train_responses['model_response'],
        df_dev_responses['model_response']
    ]).dropna().apply(preprocess_text).tolist()
    
    all_texts.extend(all_responses)
    
    # Check if model exists, train if not
    if os.path.exists(fasttext_model_path):
        print(f"Loading existing FastText model from {fasttext_model_path}")
        fasttext_model = FastText.load(fasttext_model_path)
    else:
        print("Training new FastText model...")
        fasttext_model = train_fasttext_model(all_texts, fasttext_model_path)
    
    # Generate embeddings
    print("Generating embeddings for training prompts...")
    train_embeddings = np.array([
        get_sentence_embedding(fasttext_model, text) 
        for text in tqdm(df_train_prompts['processed'])
    ])
    
    print("Generating embeddings for dev prompts...")
    dev_embeddings = np.array([
        get_sentence_embedding(fasttext_model, text) 
        for text in tqdm(df_dev_prompts['processed'])
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
    
    # Process test set
    print("Generating embeddings for test prompts...")
    test_embeddings = np.array([
        get_sentence_embedding(fasttext_model, text) 
        for text in tqdm(df_test_prompts['processed'])
    ])
    
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