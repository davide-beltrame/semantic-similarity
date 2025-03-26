#!/usr/bin/env python3
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import argparse

def load_data(use_dev=False):
    """
    Load data properly handling paths and directories.
    """
    # Get the current directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(script_dir), "data")
    dump_dir = os.path.join(os.path.dirname(script_dir), "dump")
    
    # Create dump directory if it doesn't exist
    os.makedirs(dump_dir, exist_ok=True)
    
    # Load training data
    train_prompts_path = os.path.join(data_dir, "train_prompts.csv")
    train_df = pd.read_csv(train_prompts_path)
    
    # Load target data and determine output path
    if not use_dev:  # test mode
        test_prompts_path = os.path.join(data_dir, "test_prompts.csv")
        target_df = pd.read_csv(test_prompts_path)
        output_file = os.path.join(dump_dir, "track1_test.csv")
        
        # For test data, include dev data in retrieval pool
        dev_prompts_path = os.path.join(data_dir, "dev_prompts.csv")
        dev_df = pd.read_csv(dev_prompts_path)
        combined_df = pd.concat([train_df, dev_df], ignore_index=True)
    else:  # dev mode
        dev_prompts_path = os.path.join(data_dir, "dev_prompts.csv")
        target_df = pd.read_csv(dev_prompts_path)
        output_file = os.path.join(dump_dir, "track1_dev.csv")
        
        # For dev data, only use train data as retrieval pool
        combined_df = train_df.copy()
    
    return combined_df, target_df, output_file

def build_representation(train_prompts):
    """
    Build a TF-IDF representation for prompts.
    """
    # Create vectorizer with optimal parameters for semantic similarity
    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),    # Include both unigrams and bigrams
        min_df=2,              # Ignore terms that appear in less than 2 documents
        max_df=0.9,            # Ignore terms that appear in more than 90% of documents
        max_features=10000     # Limit features to improve efficiency
    )
    
    train_vectors = vectorizer.fit_transform(train_prompts)
    print(f"TF-IDF matrix shape: {train_vectors.shape}")
    
    return vectorizer, train_vectors

def find_best_match(test_prompt, vectorizer, train_vectors, combined_df):
    """
    Find most similar prompt from retrieval pool.
    """
    test_vec = vectorizer.transform([test_prompt])
    sims = cosine_similarity(test_vec, train_vectors).flatten()
    best_idx = sims.argmax()
    return best_idx

def run_retrieval(use_dev=False):
    """Main retrieval function."""
    print(f"Running retrieval for {'dev' if use_dev else 'test'} dataset...")
    
    # 1) Load data
    combined_df, target_df, output_file = load_data(use_dev)
    print(f"Loaded {len(combined_df)} prompts in retrieval pool")
    print(f"Loaded {len(target_df)} {'dev' if use_dev else 'test'} prompts")
    
    # 2) Build representation
    vectorizer, train_vectors = build_representation(combined_df["user_prompt"])
    
    # 3) For each prompt, find best match
    results = []
    
    for i, row in target_df.iterrows():
        target_id = row["conversation_id"]
        test_prompt = row["user_prompt"]
        
        # Find best match
        best_idx = find_best_match(test_prompt, vectorizer, train_vectors, combined_df)
        best_id = combined_df.iloc[best_idx]["conversation_id"]
        
        # Store result
        results.append({
            "conversation_id": target_id,
            "response_id": best_id
        })
        
        # Print progress every 1000 prompts
        if i % 1000 == 0 and i > 0:
            print(f"Processed {i} out of {len(target_df)} prompts")
    
    # 4) Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    
    # Print statistics for dev set
    if use_dev:
        matches = sum(results_df["conversation_id"] == results_df["response_id"])
        print(f"Self-matches: {matches} out of {len(results_df)} ({matches/len(results_df)*100:.2f}%)")
    
    return output_file

def main():
    parser = argparse.ArgumentParser(description="Track 1: TF-IDF based retrieval for semantic similarity")
    parser.add_argument("--use_dev", action="store_true", help="Use dev set for evaluation")
    args = parser.parse_args()
    
    run_retrieval(args.use_dev)

if __name__ == "__main__":
    main()