import pandas as pd
import numpy as np
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def preprocess_text(text):
    """
    Enhanced preprocessing for n-grams:
      - Lowercase, strip, normalize whitespace.
    """
    if pd.isna(text):
        return ""
    text = str(text).lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def preprocess_char(text):
    """
    Character-specific preprocessing:
      - Lowercase, strip, normalize whitespace.
      - Add leading/trailing spaces to capture word boundaries.
    """
    if pd.isna(text):
        return ""
    text = str(text).lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return ' ' + text + ' '

def main():
    # Set up paths relative to this script's location
    code_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(code_dir, "../data")
    dump_dir = os.path.join(code_dir, "../dump")
    os.makedirs(dump_dir, exist_ok=True)

    # Load development data for evaluation
    df_dev_prompts = pd.read_csv(os.path.join(data_dir, "dev_prompts.csv"))
    df_dev_responses = pd.read_csv(os.path.join(data_dir, "dev_responses.csv"))
    # Load training data for retrieval pool
    df_train_prompts = pd.read_csv(os.path.join(data_dir, "train_prompts.csv"))
    df_train_responses = pd.read_csv(os.path.join(data_dir, "train_responses.csv"))
    
    # Process for both word and character representations
    df_train_prompts['processed_word'] = df_train_prompts['user_prompt'].apply(preprocess_text)
    df_dev_prompts['processed_word'] = df_dev_prompts['user_prompt'].apply(preprocess_text)
    
    df_train_prompts['processed_char'] = df_train_prompts['user_prompt'].apply(preprocess_char)
    df_dev_prompts['processed_char'] = df_dev_prompts['user_prompt'].apply(preprocess_char)
    
    # Build word-level TF-IDF vectorizer
    word_vectorizer = TfidfVectorizer(
        analyzer='word',
        ngram_range=(1, 3),  # Unigrams, bigrams, and trigrams
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        use_idf=True,
        norm='l2'
    )
    
    # Build character-level TF-IDF vectorizer (keeping your original settings)
    char_vectorizer = TfidfVectorizer(
        analyzer='char',
        ngram_range=(2, 5),
        min_df=2,
        max_df=0.9,
        sublinear_tf=True,
        use_idf=True,
        norm='l2',
        max_features=100000
    )
    
    # Fit and transform with word-level
    word_tfidf_train = word_vectorizer.fit_transform(df_train_prompts['processed_word'])
    word_tfidf_dev = word_vectorizer.transform(df_dev_prompts['processed_word'])
    
    # Fit and transform with character-level
    char_tfidf_train = char_vectorizer.fit_transform(df_train_prompts['processed_char'])
    char_tfidf_dev = char_vectorizer.transform(df_dev_prompts['processed_char'])
    
    print(f"Word TF-IDF shapes - Train: {word_tfidf_train.shape}, Dev: {word_tfidf_dev.shape}")
    print(f"Char TF-IDF shapes - Train: {char_tfidf_train.shape}, Dev: {char_tfidf_dev.shape}")
    
    # Compute similarity matrices for both representations
    word_similarity = cosine_similarity(word_tfidf_dev, word_tfidf_train)
    char_similarity = cosine_similarity(char_tfidf_dev, char_tfidf_train)
    
    # Create ensemble by weighted averaging (tunable weights)
    word_weight = 0.4  # Weight for word-level similarity
    char_weight = 0.6  # Weight for character-level similarity (keeping higher as your original performed well)
    
    # Combine similarities
    ensemble_similarity = (word_weight * word_similarity) + (char_weight * char_similarity)
    
    # Get best matches from ensemble similarity
    best_indices = np.argmax(ensemble_similarity, axis=1)
    
    # Get the corresponding training conversation_ids for the best matches
    best_train_ids = df_train_prompts.iloc[best_indices]['conversation_id'].values
    
    # Build response mappings for train and dev
    train_response_map = dict(zip(df_train_responses['conversation_id'], df_train_responses['model_response']))
    dev_response_map = dict(zip(df_dev_responses['conversation_id'], df_dev_responses['model_response']))
    
    # Compute BLEU scores: compare each dev response (true) against the retrieved train response
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
    
    # Dump the detailed results to a CSV in the dump folder
    results_df = pd.DataFrame({
        "dev_conversation_id": df_dev_prompts['conversation_id'],
        "retrieved_train_id": best_train_ids,
        "true_response": df_dev_prompts['conversation_id'].map(dev_response_map),
        "predicted_response": [train_response_map.get(rid, "") for rid in best_train_ids],
        "bleu_score": bleu_scores
    })
    output_csv = os.path.join(dump_dir, "dev_bleu_evaluation_ensemble.csv")
    results_df.to_csv(output_csv, index=False)
    
    # For test set predictions
    # Load test prompts
    df_test_prompts = pd.read_csv(os.path.join(data_dir, "test_prompts.csv"))
    df_test_prompts['processed_word'] = df_test_prompts['user_prompt'].apply(preprocess_text)
    df_test_prompts['processed_char'] = df_test_prompts['user_prompt'].apply(preprocess_char)
    
    # Transform test prompts
    word_tfidf_test = word_vectorizer.transform(df_test_prompts['processed_word'])
    char_tfidf_test = char_vectorizer.transform(df_test_prompts['processed_char'])
    
    # Compute similarities for test set
    word_test_similarity = cosine_similarity(word_tfidf_test, word_tfidf_train)
    char_test_similarity = cosine_similarity(char_tfidf_test, char_tfidf_train)
    
    # Create ensemble for test set
    test_ensemble_similarity = (word_weight * word_test_similarity) + (char_weight * char_test_similarity)
    
    # Find best matches for test prompts
    test_best_indices = np.argmax(test_ensemble_similarity, axis=1)
    test_best_train_ids = df_train_prompts.iloc[test_best_indices]['conversation_id'].values
    
    # Create test submission file
    test_results_df = pd.DataFrame({
        "conversation_id": df_test_prompts['conversation_id'],
        "response_id": test_best_train_ids
    })
    test_output_csv = os.path.join(dump_dir, "track_1_test_ensemble.csv")
    test_results_df.to_csv(test_output_csv, index=False)
    
    print(f"Average BLEU score on dev set: {avg_bleu:.5f}")
    print(f"Dev evaluation saved to: {output_csv}")
    print(f"Test predictions saved to: {test_output_csv}")

if __name__ == "__main__":
    main()