import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

def load_data(dataset="test"):
    """
    Load data properly handling paths and directories.
    """
    # Get the current directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(script_dir), "data")
    
    # Load training data
    train_prompts_path = os.path.join(data_dir, "train_prompts.csv")
    train_df = pd.read_csv(train_prompts_path)
    
    # Load target data and determine output path
    if dataset == "test":
        test_prompts_path = os.path.join(data_dir, "test_prompts.csv")
        target_df = pd.read_csv(test_prompts_path)
        output_file = os.path.join(os.path.dirname(script_dir), "track_1_test.csv")
        
        # For test data, include dev data in retrieval pool
        dev_prompts_path = os.path.join(data_dir, "dev_prompts.csv")
        dev_df = pd.read_csv(dev_prompts_path)
        combined_df = pd.concat([train_df, dev_df], ignore_index=True)
    else:  # dataset == "dev"
        dev_prompts_path = os.path.join(data_dir, "dev_prompts.csv")
        target_df = pd.read_csv(dev_prompts_path)
        output_file = os.path.join(os.path.dirname(script_dir), "track_1_dev.csv")
        
        # For dev data, only use train data as retrieval pool
        combined_df = train_df.copy()
    
    return combined_df, target_df, output_file

def build_representation(train_prompts):
    """
    Build a TF-IDF representation for prompts.
    """
    vectorizer = TfidfVectorizer(stop_words='english')
    train_vectors = vectorizer.fit_transform(train_prompts)
    return vectorizer, train_vectors

def find_best_match(test_prompt, vectorizer, train_vectors, combined_df):
    """
    Find most similar prompt from retrieval pool.
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
    
    # 4) Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

    # Print statistics
    if dataset == "dev":
        matches = sum(results_df["conversation_id"] == results_df["response_id"])
        print(f"Self-matches: {matches} out of {len(results_df)} ({matches/len(results_df)*100:.2f}%)")
    
    return output_file

if __name__ == "__main__":
    import sys
    dataset = "dev" if len(sys.argv) <= 1 else sys.argv[1]
    main(dataset)