
import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

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
    
    # Build TF-IDF representation on raw text (no preprocessing)
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
    
    tfidf_train = vectorizer.fit_transform(df_train_prompts['user_prompt'])
    tfidf_dev = vectorizer.transform(df_dev_prompts['user_prompt'])
    
    # Compute cosine similarity between dev and train prompts
    similarity_matrix = cosine_similarity(tfidf_dev, tfidf_train)
    best_indices = np.argmax(similarity_matrix, axis=1)
    
    # Retrieve the corresponding training conversation IDs for best matches
    best_train_ids = df_train_prompts.iloc[best_indices]['conversation_id'].values
    
    # Build response mappings for train and dev
    train_response_map = dict(zip(df_train_responses['conversation_id'], df_train_responses['model_response']))
    dev_response_map = dict(zip(df_dev_responses['conversation_id'], df_dev_responses['model_response']))
    
    # Compute BLEU scores for each dev prompt using retrieved train response as prediction
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
    
    # Optional: Dump detailed results into CSV in dump folder
    results_df = pd.DataFrame({
        "dev_conversation_id": df_dev_prompts['conversation_id'],
        "retrieved_train_id": best_train_ids,
        "true_response": df_dev_prompts['conversation_id'].map(dev_response_map),
        "predicted_response": [train_response_map.get(rid, "") for rid in best_train_ids],
        "bleu_score": bleu_scores
    })
    output_csv = os.path.join(dump_dir, "dev_bleu_evaluation.csv")
    results_df.to_csv(output_csv, index=False)
    
    # Print only the average BLEU score
    print(f"Average BLEU score: {avg_bleu:.5f}")

if __name__ == "__main__":
    main()
