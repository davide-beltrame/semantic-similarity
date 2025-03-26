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
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def ensure_dump_folder():
    """Ensure the dump folder exists."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dump_dir = os.path.join(os.path.dirname(script_dir), "dump")
    os.makedirs(dump_dir, exist_ok=True)
    return dump_dir

def load_data(dataset="test"):
    """Load data for retrieval."""
    # Get the current directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(script_dir), "data")
    dump_dir = ensure_dump_folder()
    
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
        output_file = os.path.join(dump_dir, "track_1_test.csv")
        
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
        output_file = os.path.join(dump_dir, "track1_bleu_evaluation.csv")
        
        # For dev data, only use train data as retrieval pool
        combined_df = train_df.copy()
        responses_df = train_responses_df
    
    return combined_df, target_df, output_file, responses_df

def build_tfidf_representation(train_prompts):
    """Build TF-IDF representation for prompts."""
    print("Building TF-IDF representation...")
    processed_prompts = [preprocess_text(prompt) for prompt in train_prompts]
    
    # Standard TF-IDF with good parameter values
    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),  # Include both unigrams and bigrams
        min_df=2,            # More aggressive filtering of rare terms
        max_df=0.8,          # More aggressive filtering of common terms
        max_features=5000    # Focus on most important features
    )
    
    train_vectors = vectorizer.fit_transform(processed_prompts)
    print(f"TF-IDF matrix shape: {train_vectors.shape}")
    
    return vectorizer, train_vectors

def extract_content_words(text):
    """Extract important content words from text."""
    words = preprocess_text(text).split()
    # Filter out stopwords (simple approach)
    stopwords = {'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what', 
                'when', 'where', 'how', 'which', 'this', 'that', 'these', 'those', 
                'then', 'to', 'of', 'in', 'for', 'with', 'by', 'at', 'from', 'is', 
                'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does',
                'did', 'can', 'could', 'will', 'would', 'should', 'i', 'you', 'he', 
                'she', 'it', 'we', 'they', 'their', 'my', 'your', 'his', 'her', 'its',
                'our', 'their'}
    content_words = [word for word in words if word not in stopwords and len(word) > 2]
    return set(content_words)

def find_best_match(test_prompt, vectorizer, train_vectors, combined_df, responses_df=None):
    """Find most similar prompt with optimized BLEU potential."""
    # Get TF-IDF similarity
    processed_test_prompt = preprocess_text(test_prompt)
    test_vec = vectorizer.transform([processed_test_prompt])
    sims = cosine_similarity(test_vec, train_vectors).flatten()
    
    # Get initial candidates - more than we need for thorough screening
    top_n = 40  # Larger pool of candidates for better chance of finding good BLEU matches
    top_indices = sims.argsort()[-top_n:][::-1]
    
    # Extract content words from test prompt
    test_content_words = extract_content_words(test_prompt)
    
    # Score the candidates on multiple dimensions
    candidate_scores = []
    
    for idx in top_indices:
        train_prompt = combined_df.iloc[idx]["user_prompt"]
        train_id = combined_df.iloc[idx]["conversation_id"]
        
        # 1. TF-IDF similarity - base metric
        tfidf_score = sims[idx]
        
        # 2. Content word overlap - critical for semantic similarity
        train_content_words = extract_content_words(train_prompt)
        word_overlap = len(test_content_words.intersection(train_content_words)) / max(1, len(test_content_words))
        
        # 3. Response characteristics (if responses are available)
        response_score = 0.5  # Default
        if responses_df is not None:
            response_rows = responses_df[responses_df["conversation_id"] == train_id]
            if not response_rows.empty:
                response = str(response_rows.iloc[0]["model_response"])
                
                # Calculate response quality factors correlated with good BLEU scores
                
                # 3a. Response length (moderate length responses tend to have better BLEU scores)
                resp_len = len(response.split())
                length_score = 0.0
                if resp_len > 20 and resp_len < 300:  # Sweet spot for response length
                    length_score = 0.8
                elif resp_len <= 20:  # Very short responses
                    length_score = 0.3
                else:  # Very long responses
                    length_score = 0.5
                
                # 3b. Response specificity (responses with content words from the prompt tend to be better)
                content_word_count = len(set(response.lower().split()).intersection(test_content_words))
                specificity_score = min(1.0, content_word_count / max(1, len(test_content_words)))
                
                # 3c. Response is not too generic (helps avoid "I don't know" type responses)
                generic_phrases = ['i dont know', 'cannot', 'sorry', 'ai', 'language model']
                generic_score = 1.0
                for phrase in generic_phrases:
                    if phrase in response.lower():
                        generic_score *= 0.8  # Penalize generic responses
                
                # Combine response factors
                response_score = (length_score + specificity_score + generic_score) / 3
        
        # Calculate final combined score for this candidate
        # Weight content word overlap more heavily as it correlates better with BLEU
        combined_score = (
            0.35 * tfidf_score +      # Base similarity
            0.45 * word_overlap +     # Content overlap (most important)
            0.20 * response_score     # Response quality
        )
        
        candidate_scores.append((idx, combined_score))
    
    # Sort by combined score and select the best
    candidate_scores.sort(key=lambda x: x[1], reverse=True)
    best_idx = candidate_scores[0][0]
    
    return best_idx

def main(dataset="test"):
    """Main retrieval function optimized for BLEU performance."""
    print(f"Running retrieval for {dataset} dataset...")
    
    # 1) Load data
    combined_df, target_df, output_file, responses_df = load_data(dataset)
    print(f"Loaded {len(combined_df)} prompts in retrieval pool")
    print(f"Loaded {len(target_df)} {dataset} prompts")
    
    # 2) Build TF-IDF representation
    vectorizer, train_vectors = build_tfidf_representation(combined_df["user_prompt"])
    
    # 3) Find best matches for each target prompt
    results = []
    
    for i, row in target_df.iterrows():
        target_id = row["conversation_id"]
        test_prompt = row["user_prompt"]
        
        # Find best match optimized for BLEU potential
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