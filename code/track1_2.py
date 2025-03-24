#!/usr/bin/env python3
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

def load_data(dataset="test"):
    """
    Load data properly handling paths and directories.
    For test: retrieval pool = train + dev; target = test prompts.
    For dev: retrieval pool = train only; target = dev prompts.
    """
    # Define directories relative to this script location
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(CURRENT_DIR, "..", "data")
    DUMP_DIR = os.path.join(CURRENT_DIR, "..", "dump")
    
    # Load training data
    train_prompts_path = os.path.join(DATA_DIR, "train_prompts.csv")
    train_df = pd.read_csv(train_prompts_path)
    
    if dataset == "test":
        test_prompts_path = os.path.join(DATA_DIR, "test_prompts.csv")
        target_df = pd.read_csv(test_prompts_path)
        output_file = os.path.join(DUMP_DIR, "track1_2_test.csv")
        
        # For test, include dev data in the retrieval pool
        dev_prompts_path = os.path.join(DATA_DIR, "dev_prompts.csv")
        dev_df = pd.read_csv(dev_prompts_path)
        combined_df = pd.concat([train_df, dev_df], ignore_index=True)
    else:  # dataset == "dev"
        dev_prompts_path = os.path.join(DATA_DIR, "dev_prompts.csv")
        target_df = pd.read_csv(dev_prompts_path)
        output_file = os.path.join(DUMP_DIR, "track1_2_dev.csv")
        
        # For dev, use only train data as the retrieval pool
        combined_df = train_df.copy()
    
    return combined_df, target_df, output_file

def build_representation(prompts):
    """
    Build a TF-IDF representation for prompts.
    """
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform(prompts)
    return vectorizer, vectors

def find_best_match(test_prompt, vectorizer, train_vectors):
    """
    Find the most similar prompt from the retrieval pool.
    """
    test_vec = vectorizer.transform([test_prompt])
    sims = cosine_similarity(test_vec, train_vectors)
    best_idx = sims.argmax()
    return best_idx

def main(dataset="test"):
    print(f"Running retrieval for {dataset} dataset...")
    
    # 1) Load data
    combined_df, target_df, output_file = load_data(dataset)
    print(f"Loaded {len(combined_df)} prompts in retrieval pool")
    print(f"Loaded {len(target_df)} {dataset} prompts")
    
    # 2) Build TF-IDF representation on the combined pool
    vectorizer, train_vectors = build_representation(combined_df["user_prompt"])
    
    # 3) For each target prompt, find the best match in the retrieval pool
    results = []
    for i, row in target_df.iterrows():
        target_id = row["conversation_id"]
        test_prompt = row["user_prompt"]
        
        best_idx = find_best_match(test_prompt, vectorizer, train_vectors)
        best_id = combined_df.iloc[best_idx]["conversation_id"]
        
        results.append({
            "conversation_id": target_id,
            "response_id": best_id
        })
    
    # 4) Save results to the dump folder with consistent naming
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

    # For dev evaluation, print self-match statistics
    if dataset == "dev":
        matches = (results_df["conversation_id"] == results_df["response_id"]).sum()
        pct = (matches / len(results_df)) * 100
        print(f"Self-matches: {matches} out of {len(results_df)} ({pct:.2f}%)")
    
    return output_file

if __name__ == "__main__":
    import sys
    dataset = "dev" if len(sys.argv) <= 1 else sys.argv[1]
    main(dataset)
