import pandas as pd
import numpy as np
import os
import re
from collections import Counter
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def preprocess_text(text):
    """Simple preprocessing: lowercase, remove punctuation, normalize whitespace."""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

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
        output_file = os.path.join(os.path.dirname(script_dir), "track_2_test.csv")
        
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
        output_file = os.path.join(os.path.dirname(script_dir), "track_2_dev.csv")
        
        # For dev data, only use train data as retrieval pool
        combined_df = train_df.copy()
        responses_df = train_responses_df
    
    return combined_df, target_df, output_file, responses_df

def build_representations(train_prompts):
    """Build both TF-IDF and LSI representations."""
    print("Building text representations...")
    
    # Preprocess all prompts
    processed_prompts = [preprocess_text(prompt) for prompt in train_prompts]
    
    # 1. Create TF-IDF representation
    print("Computing TF-IDF matrix...")
    tfidf_vectorizer = TfidfVectorizer(
        analyzer='word',
        stop_words='english',
        ngram_range=(1, 2),  # Use unigrams and bigrams
        min_df=2,            # Ignore terms appearing in less than 2 documents
        max_df=0.9,          # Ignore terms appearing in more than 90% of documents
        max_features=20000   # Limit vocabulary size for efficiency
    )
    
    tfidf_matrix = tfidf_vectorizer.fit_transform(processed_prompts)
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    
    # 2. Apply LSI (Latent Semantic Indexing) to create distributed representation
    print("Creating LSI representation...")
    n_components = 150  # Dimensionality of LSI space
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    lsi_matrix = svd.fit_transform(tfidf_matrix)
    print(f"LSI matrix shape: {lsi_matrix.shape}")
    
    # Explained variance as diagnostic
    explained_variance = svd.explained_variance_ratio_.sum()
    print(f"Explained variance: {explained_variance:.4f}")
    
    return {
        'tfidf_vectorizer': tfidf_vectorizer,
        'tfidf_matrix': tfidf_matrix,
        'svd': svd, 
        'lsi_matrix': lsi_matrix,
        'processed_prompts': processed_prompts
    }

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

def find_matching_responses(test_prompt, combined_df, representations, responses_df=None):
    """Find best matching response using multiple criteria."""
    # Extract components from representations
    tfidf_vectorizer = representations['tfidf_vectorizer']
    tfidf_matrix = representations['tfidf_matrix']
    svd = representations['svd']
    lsi_matrix = representations['lsi_matrix']
    
    # Process test prompt
    processed_test = preprocess_text(test_prompt)
    
    # 1. Get TF-IDF representation of test prompt
    test_tfidf = tfidf_vectorizer.transform([processed_test])
    
    # 2. Transform to LSI space
    test_lsi = svd.transform(test_tfidf)
    
    # 3. Calculate LSI similarity
    lsi_similarities = cosine_similarity(test_lsi, lsi_matrix)[0]
    
    # 4. Calculate TF-IDF similarity directly
    tfidf_similarities = cosine_similarity(test_tfidf, tfidf_matrix)[0]
    
    # 5. Get content words from test prompt
    test_content_words = extract_content_words(test_prompt)
    test_length = len(test_prompt.split())
    
    # Select top candidates based on LSI similarity
    top_n = 40  # Number of candidates to consider
    top_indices = lsi_similarities.argsort()[-top_n:][::-1]
    
    # Create more detailed scores for top candidates
    candidate_scores = []
    
    for idx in top_indices:
        train_prompt = combined_df.iloc[idx]["user_prompt"]
        train_id = combined_df.iloc[idx]["conversation_id"]
        
        # Basic similarity scores
        lsi_sim = lsi_similarities[idx]
        tfidf_sim = tfidf_similarities[idx]
        
        # Content word overlap
        train_content_words = extract_content_words(train_prompt)
        word_overlap = 0
        if len(test_content_words) > 0:
            word_overlap = len(test_content_words.intersection(train_content_words)) / len(test_content_words)
        
        # Length similarity
        train_length = len(train_prompt.split())
        length_ratio = min(test_length, train_length) / max(test_length, train_length) if max(test_length, train_length) > 0 else 0
        
        # Response quality assessment
        response_quality = 0.5  # Default
        
        if responses_df is not None:
            response_rows = responses_df[responses_df["conversation_id"] == train_id]
            if not response_rows.empty:
                response = str(response_rows.iloc[0]["model_response"])
                
                # Length-based quality score
                resp_len = len(response.split())
                if 20 <= resp_len <= 300:  # Ideal length
                    length_score = 0.9
                elif resp_len < 20:  # Too short
                    length_score = 0.3
                else:  # Too long
                    length_score = 0.6
                
                # Check for generic responses
                generic_phrases = ['i dont know', 'cannot', 'sorry', 'ai', 'language model']
                generic_penalty = 1.0
                for phrase in generic_phrases:
                    if phrase in response.lower():
                        generic_penalty *= 0.7
                
                # Check for content word presence
                content_matches = len(set(response.lower().split()).intersection(test_content_words))
                content_score = min(1.0, content_matches / max(1, len(test_content_words)))
                
                # Final response quality score
                response_quality = (length_score + generic_penalty + content_score) / 3
        
        # Combined score with tuned weights
        combined_score = (
            0.30 * lsi_sim +         # LSI semantic similarity
            0.20 * tfidf_sim +       # Direct TF-IDF similarity
            0.25 * word_overlap +    # Content word overlap
            0.15 * length_ratio +    # Length similarity
            0.10 * response_quality  # Response quality
        )
        
        candidate_scores.append((idx, combined_score))
    
    # Sort by combined score
    candidate_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Return the best match
    best_idx = candidate_scores[0][0]
    return best_idx

def main(dataset="test"):
    """Main retrieval function for Track 2."""
    print(f"Running improved Track 2 retrieval for {dataset} dataset...")
    
    # 1) Load data
    combined_df, target_df, output_file, responses_df = load_data(dataset)
    print(f"Loaded {len(combined_df)} prompts in retrieval pool")
    print(f"Loaded {len(target_df)} {dataset} prompts")
    
    # 2) Build representations
    representations = build_representations(combined_df["user_prompt"])
    
    # 3) Find best matches for each target prompt
    results = []
    
    for i, row in target_df.iterrows():
        target_id = row["conversation_id"]
        test_prompt = row["user_prompt"]
        
        # Find best match
        best_idx = find_matching_responses(test_prompt, combined_df, representations, responses_df)
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