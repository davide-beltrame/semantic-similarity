import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import re

def preprocess_text(text):
    """Lowercase, remove punctuation, and normalize whitespace."""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def ensure_dump_folder():
    """Ensure the dump folder exists."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dump_dir = os.path.join(os.path.dirname(script_dir), "dump")
    os.makedirs(dump_dir, exist_ok=True)
    return dump_dir

def load_data(dataset="test"):
    """
    Load data from the data folder.
    
    Files expected:
      - train_prompts.csv, train_responses.csv
      - dev_prompts.csv, dev_responses.csv
      - test_prompts.csv
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(script_dir), "data")
    dump_dir = ensure_dump_folder()
    
    # Load training data
    train_df = pd.read_csv(os.path.join(data_dir, "train_prompts.csv"))
    train_responses_df = pd.read_csv(os.path.join(data_dir, "train_responses.csv"))
    
    if dataset == "test":
        target_df = pd.read_csv(os.path.join(data_dir, "test_prompts.csv"))
        # Retrieval pool: train + dev
        dev_df = pd.read_csv(os.path.join(data_dir, "dev_prompts.csv"))
        combined_df = pd.concat([train_df, dev_df], ignore_index=True)
        # Combine responses if available
        try:
            dev_responses_df = pd.read_csv(os.path.join(data_dir, "dev_responses.csv"))
            responses_df = pd.concat([train_responses_df, dev_responses_df], ignore_index=True)
        except FileNotFoundError:
            responses_df = train_responses_df
        target_responses_df = None
        output_file = os.path.join(dump_dir, "track_1_test.csv")
    
    elif dataset == "train":
        # Target: train; retrieval pool: train only
        target_df = train_df.copy()
        combined_df = train_df.copy()
        responses_df = train_responses_df.copy()
        target_responses_df = train_responses_df.copy()
        output_file = os.path.join(dump_dir, "track_1_train.csv")
    
    else:  # dataset == "dev"
        target_df = pd.read_csv(os.path.join(data_dir, "dev_prompts.csv"))
        combined_df = train_df.copy()  # use training data as retrieval pool
        responses_df = train_responses_df.copy()
        try:
            target_responses_df = pd.read_csv(os.path.join(data_dir, "dev_responses.csv"))
        except FileNotFoundError:
            target_responses_df = None
        # For evaluation, name the file so tester.py can find it
        output_file = os.path.join(dump_dir, "track1_simple_evaluation.csv")
    
    return combined_df, target_df, output_file, responses_df, target_responses_df

def build_tfidf_representation(prompts):
    """Build a TF-IDF vectorizer and transform the prompts."""
    processed = [preprocess_text(p) for p in prompts]
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2), min_df=3, max_df=0.8, max_features=10000)
    tfidf_matrix = vectorizer.fit_transform(processed)
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    return vectorizer, tfidf_matrix

def find_best_match(test_prompt, vectorizer, pool_vectors):
    """Return the index of the best matching prompt (highest cosine similarity)."""
    test_vec = vectorizer.transform([preprocess_text(test_prompt)])
    sims = cosine_similarity(test_vec, pool_vectors).flatten()
    return sims.argmax()

def main(dataset="test"):
    print(f"Running retrieval for {dataset} dataset...")
    combined_df, target_df, output_file, responses_df, target_responses_df = load_data(dataset)
    print(f"Retrieval pool size: {len(combined_df)} prompts")
    print(f"Target set size: {len(target_df)} prompts")
    
    vectorizer, pool_vectors = build_tfidf_representation(combined_df["user_prompt"])
    
    results = []
    for i, row in target_df.iterrows():
        target_id = row["conversation_id"]
        test_prompt = row["user_prompt"]
        best_idx = find_best_match(test_prompt, vectorizer, pool_vectors)
        best_id = combined_df.iloc[best_idx]["conversation_id"]
        
        result = {"conversation_id": target_id, "response_id": best_id}
        # If reference responses are available, add them for BLEU evaluation.
        if dataset in ["dev", "train"] and target_responses_df is not None:
            # Get reference response for the target prompt
            ref_row = target_responses_df[target_responses_df["conversation_id"] == target_id]
            result["model_response"] = ref_row.iloc[0]["model_response"] if not ref_row.empty else ""
            # Get retrieved response from the retrieval pool
            pool_row = responses_df[responses_df["conversation_id"] == best_id]
            result["retrieved_response"] = pool_row.iloc[0]["model_response"] if not pool_row.empty else ""
        results.append(result)
        
        if (i+1) % 1000 == 0:
            print(f"Processed {i+1} out of {len(target_df)} prompts")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    
    if dataset in ["dev", "train"]:
        # Print self-match stats (how many times the target matches its own retrieval)
        self_matches = (results_df["conversation_id"] == results_df["response_id"]).sum()
        pct = self_matches / len(results_df) * 100
        print(f"Self-matches: {self_matches} out of {len(results_df)} ({pct:.2f}%)")
    
    return output_file

if __name__ == "__main__":
    import sys
    # Allow dataset to be chosen from the command line: "train", "dev", or "test"
    dataset = "dev" if len(sys.argv) <= 1 else sys.argv[1]
    main(dataset)
