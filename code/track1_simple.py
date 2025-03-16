import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import re
from collections import Counter

def preprocess_text(text):
    """Preprocess text for better matching."""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def calculate_jaccard_similarity(set1, set2):
    """Calculate Jaccard similarity between two sets."""
    if not set1 or not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0

def load_data(dataset="test"):
    """Load data for retrieval."""
    # Get the current directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(script_dir), "data")
    
    # Load training data
    train_prompts_path = os.path.join(data_dir, "train_prompts.csv")
    train_df = pd.read_csv(train_prompts_path)
    
    # Load train responses
    train_responses_path = os.path.join(data_dir, "train_responses.csv")
    try:
        train_responses_df = pd.read_csv(train_responses_path)
        print(f"Loaded {len(train_responses_df)} training responses")
    except FileNotFoundError:
        print("Warning: Training responses file not found")
        train_responses_df = None
    
    # Load target data and determine output path
    if dataset == "test":
        test_prompts_path = os.path.join(data_dir, "test_prompts.csv")
        target_df = pd.read_csv(test_prompts_path)
        output_file = os.path.join(os.path.dirname(script_dir), "track_1_test.csv")
        
        # For test data, include dev data in retrieval pool
        dev_prompts_path = os.path.join(data_dir, "dev_prompts.csv")
        dev_df = pd.read_csv(dev_prompts_path)
        combined_df = pd.concat([train_df, dev_df], ignore_index=True)
        
        # Load dev responses if available
        dev_responses_path = os.path.join(data_dir, "dev_responses.csv")
        try:
            dev_responses_df = pd.read_csv(dev_responses_path)
            if train_responses_df is not None:
                responses_df = pd.concat([train_responses_df, dev_responses_df], ignore_index=True)
            else:
                responses_df = dev_responses_df
            print(f"Loaded {len(dev_responses_df)} dev responses")
        except FileNotFoundError:
            print("Warning: Dev responses file not found")
            responses_df = train_responses_df
            
    else:  # dataset == "dev"
        dev_prompts_path = os.path.join(data_dir, "dev_prompts.csv")
        target_df = pd.read_csv(dev_prompts_path)
        output_file = os.path.join(os.path.dirname(script_dir), "track_1_dev.csv")
        
        # For dev data, only use train data as retrieval pool
        combined_df = train_df.copy()
        responses_df = train_responses_df
    
    return combined_df, target_df, output_file, responses_df

def build_representation(train_prompts):
    """Build TF-IDF representation for prompts."""
    print("Building TF-IDF representation...")
    processed_prompts = [preprocess_text(prompt) for prompt in train_prompts]
    
    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),  # Include both unigrams and bigrams
        min_df=2,            # Ignore rare terms
        max_df=0.9,          # Ignore very common terms
        max_features=10000   # Limit features
    )
    
    train_vectors = vectorizer.fit_transform(processed_prompts)
    print(f"TF-IDF matrix shape: {train_vectors.shape}")
    
    return vectorizer, train_vectors, processed_prompts

def find_best_match(test_prompt, vectorizer, train_vectors, combined_df, responses_df=None):
    """Find most similar prompt with enhanced similarity criteria."""
    processed_test_prompt = preprocess_text(test_prompt)
    test_vec = vectorizer.transform([processed_test_prompt])
    
    # Get TF-IDF similarity scores
    sims = cosine_similarity(test_vec, train_vectors).flatten()
    
    # Get top candidates
    top_n = 15
    top_indices = sims.argsort()[-top_n:][::-1]
    
    # Calculate enhanced similarity scores
    best_score = -1
    best_idx = top_indices[0]  # Default to highest TF-IDF
    
    for idx in top_indices:
        train_prompt = combined_df.iloc[idx]["user_prompt"]
        train_id = combined_df.iloc[idx]["conversation_id"]
        
        # 1. TF-IDF similarity
        tfidf_score = sims[idx]
        
        # 2. Length similarity
        test_len = len(str(test_prompt).split())
        train_len = len(str(train_prompt).split())
        len_ratio = min(test_len, train_len) / max(test_len, train_len) if max(test_len, train_len) > 0 else 0
        
        # 3. Word overlap
        test_words = set(preprocess_text(test_prompt).split())
        train_words = set(preprocess_text(train_prompt).split())
        word_overlap = calculate_jaccard_similarity(test_words, train_words)
        
        # 4. Response quality (if available)
        response_quality = 0.5  # Default
        if responses_df is not None:
            response_rows = responses_df[responses_df["conversation_id"] == train_id]
            if not response_rows.empty:
                response = response_rows.iloc[0]["model_response"]
                # Favor longer responses (up to a point)
                resp_len = len(str(response).split())
                response_quality = min(1.0, resp_len / 200)
        
        # Combine scores with weights
        combined_score = (
            0.40 * tfidf_score +       # TF-IDF similarity
            0.20 * len_ratio +         # Length similarity 
            0.30 * word_overlap +      # Word overlap
            0.10 * response_quality    # Response quality
        )
        
        if combined_score > best_score:
            best_score = combined_score
            best_idx = idx
    
    return best_idx

def main(dataset="test"):
    """Main retrieval function."""
    print(f"Running retrieval for {dataset} dataset...")
    
    # 1) Load data
    combined_df, target_df, output_file, responses_df = load_data(dataset)
    print(f"Loaded {len(combined_df)} prompts in retrieval pool")
    print(f"Loaded {len(target_df)} {dataset} prompts")
    
    # 2) Build TF-IDF representation
    vectorizer, train_vectors, _ = build_representation(combined_df["user_prompt"])
    
    # 3) Find best matches for each target prompt
    results = []
    
    for i, row in target_df.iterrows():
        target_id = row["conversation_id"]
        test_prompt = row["user_prompt"]
        
        # Find best match
        best_idx = find_best_match(test_prompt, vectorizer, train_vectors, combined_df, responses_df)
        best_id = combined_df.iloc[best_idx]["conversation_id"]
        
        # Store result
        results.append({
            "conversation_id": target_id,
            "response_id": best_id
        })
        
        # Print progress
        if i % 1000 == 0 and i > 0:
            print(f"Processed {i} out of {len(target_df)} prompts")
    
    # 4) Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    
    # Print statistics for dev set
    if dataset == "dev":
        matches = sum(results_df["conversation_id"] == results_df["response_id"])
        print(f"Self-matches: {matches} out of {len(results_df)} ({matches/len(results_df)*100:.2f}%)")
    
    return output_file

if __name__ == "__main__":
    import sys
    dataset = "dev" if len(sys.argv) <= 1 else sys.argv[1]
    main(dataset)