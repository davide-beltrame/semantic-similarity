import pandas as pd
import numpy as np
import re
import os
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Response filtering
MIN_RESPONSE_LENGTH = 3  # Min number of characters for valid responses

def preprocess_text(text, method="with_boundaries"):
    """
    Enhanced preprocessing for character n-grams:
    - Lowercase, strip, normalize whitespace.
    - Add leading/trailing spaces to capture word boundaries.
    """
    if pd.isna(text):
        return ""
    
    text = str(text).lower().strip()
    text = re.sub(r'\s+', ' ', text)
    
    if method == "with_boundaries":
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

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Track 1: TF-IDF Character N-gram with Response Filtering")
    parser.add_argument("--no_filter", action="store_true", help="Don't filter invalid responses")
    parser.add_argument("--output_dir", type=str, default=None, 
                        help="Directory for output files (default: parent directory of script)")
    args = parser.parse_args()
    
    filter_responses = not args.no_filter
    
    # Set up paths relative to this script's location
    code_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(code_dir, "data")
    
    # Set output directory to parent directory by default
    parent_dir = os.path.dirname(code_dir)  # Get parent directory of code_dir
    output_dir = args.output_dir if args.output_dir else parent_dir
    os.makedirs(output_dir, exist_ok=True)
        
    # Load development data for evaluation
    df_dev_prompts = pd.read_csv(os.path.join(data_dir, "dev_prompts.csv"))
    df_dev_responses = pd.read_csv(os.path.join(data_dir, "dev_responses.csv"))
    
    # Load training data for retrieval pool
    df_train_prompts = pd.read_csv(os.path.join(data_dir, "train_prompts.csv"))
    df_train_responses = pd.read_csv(os.path.join(data_dir, "train_responses.csv"))
    
    # Apply response filtering if requested
    if filter_responses:
        df_train_prompts, df_train_responses = filter_invalid_responses(df_train_prompts, df_train_responses)
    
    # Apply preprocessing using the best configuration
    preprocessing_method = "with_boundaries"
    df_train_prompts['processed'] = df_train_prompts['user_prompt'].apply(
        lambda x: preprocess_text(x, preprocessing_method))
    df_dev_prompts['processed'] = df_dev_prompts['user_prompt'].apply(
        lambda x: preprocess_text(x, preprocessing_method))
    
    # Build TF-IDF vectorizer with best configuration
    vectorizer = TfidfVectorizer(
        analyzer='char',
        ngram_range=(2, 4),
        min_df=3,
        max_df=0.7,
        sublinear_tf=True,
        use_idf=True,
        norm='l1',
        max_features=100000
    )
    
    # Fit vectorizer on training prompts and transform dev prompts
    tfidf_train = vectorizer.fit_transform(df_train_prompts['processed'])
    tfidf_dev = vectorizer.transform(df_dev_prompts['processed'])
    
    # For each dev prompt, find the best matching train prompt via cosine similarity
    similarity_matrix = cosine_similarity(tfidf_dev, tfidf_train)
    best_indices = np.argmax(similarity_matrix, axis=1)
    
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
    
    # Create suffix for output files based on filtering
    suffix = "_filtered" if filter_responses else ""
    
    # Dump the detailed results to a CSV in the output folder
    results_df = pd.DataFrame({
        "dev_conversation_id": df_dev_prompts['conversation_id'],
        "retrieved_train_id": best_train_ids,
        "true_response": df_dev_prompts['conversation_id'].map(dev_response_map),
        "predicted_response": [train_response_map.get(rid, "") for rid in best_train_ids],
        "bleu_score": bleu_scores
    })
    
    dev_output_csv = os.path.join(output_dir, f"track_1_dev{suffix}.csv")
    results_df.to_csv(dev_output_csv, index=False)
    
    # Print only the average BLEU score
    print(f"Average BLEU score on dev set: {avg_bleu:.5f}")
    
    # Process test data
    df_test_prompts = pd.read_csv(os.path.join(data_dir, "test_prompts.csv"))
    df_test_prompts['processed'] = df_test_prompts['user_prompt'].apply(
        lambda x: preprocess_text(x, preprocessing_method))
    
    # Combine train and dev for testing
    candidate_prompts = pd.concat([df_train_prompts, df_dev_prompts], ignore_index=True)
    
    # Re-fit vectorizer on the combined data
    tfidf_candidates = vectorizer.fit_transform(candidate_prompts['processed'])
    tfidf_test = vectorizer.transform(df_test_prompts['processed'])
    
    # Compute similarities between test and candidate prompts
    test_similarity_matrix = cosine_similarity(tfidf_test, tfidf_candidates)
    test_best_indices = np.argmax(test_similarity_matrix, axis=1)
    test_best_candidate_ids = candidate_prompts.iloc[test_best_indices]['conversation_id'].values
    
    # Create and save test submission file
    test_results_df = pd.DataFrame({
        "conversation_id": df_test_prompts['conversation_id'],
        "response_id": test_best_candidate_ids
    })
    
    # Save as track_1_test.csv in the output directory
    test_output_csv = os.path.join(output_dir, "track_1_test.csv")
    test_results_df.to_csv(test_output_csv, index=False)
    
    print(f"Test predictions saved to: {test_output_csv}")

if __name__ == "__main__":
    main()