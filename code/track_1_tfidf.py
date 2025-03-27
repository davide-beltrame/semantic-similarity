import pandas as pd
import numpy as np
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def preprocess_text(text):
    """
    Enhanced preprocessing for character n-grams:
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
    
    # Preprocess the prompts (only for retrieval purposes)
    df_train_prompts['processed'] = df_train_prompts['user_prompt'].apply(preprocess_text)
    df_dev_prompts['processed']   = df_dev_prompts['user_prompt'].apply(preprocess_text)
    
    # Build TF-IDF vectorizer (enhanced settings)
    vectorizer = TfidfVectorizer(
        analyzer='char',
        ngram_range=(2, 5),
        min_df=2,
        max_df=0.9,
        sublinear_tf=True,
        use_idf=True,
        norm='l2',
        max_features=100000
    )
    
    # Fit vectorizer on training prompts and transform dev prompts
    tfidf_train = vectorizer.fit_transform(df_train_prompts['processed'])
    tfidf_dev   = vectorizer.transform(df_dev_prompts['processed'])
    
    # For each dev prompt, find the best matching train prompt via cosine similarity
    similarity_matrix = cosine_similarity(tfidf_dev, tfidf_train)
    best_indices = np.argmax(similarity_matrix, axis=1)
    
    # Get the corresponding training conversation_ids for the best matches
    best_train_ids = df_train_prompts.iloc[best_indices]['conversation_id'].values
    
    # Build response mappings for train and dev
    train_response_map = dict(zip(df_train_responses['conversation_id'], df_train_responses['model_response']))
    dev_response_map   = dict(zip(df_dev_responses['conversation_id'], df_dev_responses['model_response']))
    
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
    
    # Dump the detailed results (optional) to a CSV in the dump folder
    results_df = pd.DataFrame({
        "dev_conversation_id": df_dev_prompts['conversation_id'],
        "retrieved_train_id": best_train_ids,
        "true_response": df_dev_prompts['conversation_id'].map(dev_response_map),
        "predicted_response": [train_response_map.get(rid, "") for rid in best_train_ids],
        "bleu_score": bleu_scores
    })
    output_csv = os.path.join(dump_dir, "track_1_dev.csv")
    results_df.to_csv(output_csv, index=False)
    
    # Print only the average BLEU score
    print(f"Average BLEU score: {avg_bleu:.5f}")
    
    # Process test data
    df_test_prompts = pd.read_csv(os.path.join(data_dir, "test_prompts.csv"))
    df_test_prompts['processed'] = df_test_prompts['user_prompt'].apply(preprocess_text)
    
    # Transform test prompts using the same vectorizer
    tfidf_test = vectorizer.transform(df_test_prompts['processed'])
    
    # Compute similarities between test and train prompts
    test_similarity_matrix = cosine_similarity(tfidf_test, tfidf_train)
    test_best_indices = np.argmax(test_similarity_matrix, axis=1)
    test_best_train_ids = df_train_prompts.iloc[test_best_indices]['conversation_id'].values
    
    # Create and save test submission file
    test_results_df = pd.DataFrame({
        "conversation_id": df_test_prompts['conversation_id'],
        "response_id": test_best_train_ids
    })
    test_output_csv = os.path.join(dump_dir, "track_1_test.csv")
    test_results_df.to_csv(test_output_csv, index=False)
    print(f"Test predictions saved to: {test_output_csv}")

if __name__ == "__main__":
    main()