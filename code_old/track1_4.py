import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import re
from collections import Counter

def preprocess_text(text):
    """Simple preprocessing: lowercase, remove punctuation, normalize whitespace."""
    text = str(text).lower()
    text = re.sub(r'[^\w\s?]', ' ', text)  # Keep question marks
    return re.sub(r'\s+', ' ', text).strip()

def ensure_dump_folder():
    """Ensure the dump folder exists."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dump_dir = os.path.join(os.path.dirname(script_dir), "dump")
    os.makedirs(dump_dir, exist_ok=True)
    return dump_dir

def load_data(dataset="test"):
    """Load data from the data folder."""
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
    else:  # dataset == "dev" or "train"
        if dataset == "train":
            target_df = train_df.copy()
            target_responses_df = train_responses_df.copy()
            output_file = os.path.join(dump_dir, "track_1_train.csv")
        else:  # dev
            target_df = pd.read_csv(os.path.join(data_dir, "dev_prompts.csv"))
            try:
                target_responses_df = pd.read_csv(os.path.join(data_dir, "dev_responses.csv"))
            except FileNotFoundError:
                target_responses_df = None
            output_file = os.path.join(dump_dir, "track1_4_evaluation.csv")
        
        combined_df = train_df.copy()  # use training data as retrieval pool
        responses_df = train_responses_df.copy()
    
    return combined_df, target_df, output_file, responses_df, target_responses_df

def extract_important_words(text, stopwords):
    """Extract important words (non-stopwords) from text."""
    words = str(text).lower().split()
    return [w for w in words if w not in stopwords and len(w) > 2]

def build_tfidf_representation(prompts):
    """Build an enhanced TF-IDF representation with tuned parameters."""
    processed = [preprocess_text(p) for p in prompts]
    
    # TF-IDF with optimized parameters
    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),
        min_df=3, 
        max_df=0.85,
        max_features=8000
    )
    
    tfidf_matrix = vectorizer.fit_transform(processed)
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    
    return vectorizer, tfidf_matrix, processed

def find_best_match(test_prompt, vectorizer, pool_vectors, processed_prompts, combined_df, responses_df=None):
    """Find the best matching prompt using multiple criteria."""
    # Process test prompt
    processed_test = preprocess_text(test_prompt)
    
    # Calculate TF-IDF similarity
    test_vec = vectorizer.transform([processed_test])
    sims = cosine_similarity(test_vec, pool_vectors).flatten()
    
    # Get top candidates
    top_n = 30  # Consider fewer candidates for speed
    top_indices = sims.argsort()[-top_n:][::-1]
    
    # For very fast operation, just return the top TF-IDF match
    if responses_df is None:
        return top_indices[0]
    
    # Common stopwords for content word extraction
    stopwords = {'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what', 
                'when', 'where', 'how', 'which', 'this', 'that', 'these', 'those', 
                'then', 'to', 'of', 'in', 'for', 'with', 'by', 'at', 'from', 'is'}
    
    # Extract important words from test prompt
    test_words = set(extract_important_words(test_prompt, stopwords))
    test_len = len(test_prompt.split())
    
    # Score candidates with additional criteria
    best_score = -1
    best_idx = top_indices[0]  # Default to highest TF-IDF
    
    for idx in top_indices:
        # Start with TF-IDF score
        score = sims[idx] * 0.6  # Base: 60% TF-IDF
        
        # Get train prompt and response
        train_prompt = combined_df.iloc[idx]["user_prompt"]
        train_id = combined_df.iloc[idx]["conversation_id"]
        
        # Calculate content word overlap (faster than Jaccard)
        train_words = set(extract_important_words(train_prompt, stopwords))
        if test_words:
            overlap = len(test_words.intersection(train_words)) / len(test_words)
            score += overlap * 0.3  # 30% content overlap
        
        # Add length similarity (minimal computation)
        train_len = len(train_prompt.split())
        len_ratio = min(test_len, train_len) / max(test_len, train_len) if max(test_len, train_len) > 0 else 0
        score += len_ratio * 0.1  # 10% length similarity
        
        # Add response quality
        response_rows = responses_df[responses_df["conversation_id"] == train_id]
        if not response_rows.empty:
            response = str(response_rows.iloc[0]["model_response"])
            resp_len = len(response.split())
            
            # Simple but effective heuristic - favor medium length responses
            if 30 <= resp_len <= 200:
                score += 0.05  # Bonus for optimal length
            
            # Quick check for generic responses
            if any(phrase in response.lower() for phrase in ['i dont know', 'cannot', 'sorry', 'as an ai']):
                score -= 0.05  # Penalty for generic responses
        
        if score > best_score:
            best_score = score
            best_idx = idx
    
    return best_idx

def main(dataset="test"):
    print(f"Running retrieval for {dataset} dataset...")
    combined_df, target_df, output_file, responses_df, target_responses_df = load_data(dataset)
    print(f"Retrieval pool size: {len(combined_df)} prompts")
    print(f"Target set size: {len(target_df)} prompts")
    
    # Build TF-IDF representation
    vectorizer, pool_vectors, processed_prompts = build_tfidf_representation(combined_df["user_prompt"])
    
    # Process batches for speed
    results = []
    batch_size = 100  # Process in batches for better performance
    
    for start_idx in range(0, len(target_df), batch_size):
        end_idx = min(start_idx + batch_size, len(target_df))
        batch = target_df.iloc[start_idx:end_idx]
        
        for i, row in batch.iterrows():
            target_id = row["conversation_id"]
            test_prompt = row["user_prompt"]
            
            # Find best match
            best_idx = find_best_match(test_prompt, vectorizer, pool_vectors, processed_prompts, combined_df, responses_df)
            best_id = combined_df.iloc[best_idx]["conversation_id"]
            
            result = {"conversation_id": target_id, "response_id": best_id}
            # Add responses if available
            if dataset in ["dev", "train"] and target_responses_df is not None:
                ref_row = target_responses_df[target_responses_df["conversation_id"] == target_id]
                result["model_response"] = ref_row.iloc[0]["model_response"] if not ref_row.empty else ""
                pool_row = responses_df[responses_df["conversation_id"] == best_id]
                result["retrieved_response"] = pool_row.iloc[0]["model_response"] if not pool_row.empty else ""
            
            results.append(result)
        
        # Print progress after each batch
        print(f"Processed {min(end_idx, len(target_df))} out of {len(target_df)} prompts")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    
    if dataset in ["dev", "train"]:
        # Print self-match stats
        self_matches = (results_df["conversation_id"] == results_df["response_id"]).sum()
        pct = self_matches / len(results_df) * 100
        print(f"Self-matches: {self_matches} out of {len(results_df)} ({pct:.2f}%)")
    
    return output_file

if __name__ == "__main__":
    import sys
    dataset = "dev" if len(sys.argv) <= 1 else sys.argv[1]
    main(dataset)