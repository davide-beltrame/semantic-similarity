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

def build_embeddings(prompts):
    """Build distributed embeddings using TF-IDF + LSA (Latent Semantic Analysis)."""
    print("Building distributed representations...")
    
    # 1. Preprocess all prompts
    processed_prompts = [preprocess_text(prompt) for prompt in prompts]
    
    # 2. Create TF-IDF matrix
    print("Computing TF-IDF matrix...")
    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2, 
        max_df=0.9
    )
    tfidf_matrix = vectorizer.fit_transform(processed_prompts)
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    
    # 3. Apply SVD for dimensionality reduction (this creates distributed representations)
    print("Applying SVD for dimensionality reduction...")
    n_components = 300  # Number of dimensions for our embeddings
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    distributed_embeddings = svd.fit_transform(tfidf_matrix)
    print(f"Distributed embeddings shape: {distributed_embeddings.shape}")
    
    # 4. Normalize embeddings for better cosine similarity
    row_norms = np.linalg.norm(distributed_embeddings, axis=1)
    normalized_embeddings = distributed_embeddings / row_norms[:, np.newaxis]
    
    return {
        'vectorizer': vectorizer,
        'svd': svd,
        'embeddings': normalized_embeddings,
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

def find_best_match(test_prompt, embedding_data, combined_df, responses_df=None):
    """Find most similar prompt using distributed embeddings."""
    # Get components from embedding data
    vectorizer = embedding_data['vectorizer']
    svd = embedding_data['svd']
    embeddings = embedding_data['embeddings']
    
    # Transform test prompt to get its embedding
    processed_test_prompt = preprocess_text(test_prompt)
    test_tfidf = vectorizer.transform([processed_test_prompt])
    test_embedding = svd.transform(test_tfidf)
    
    # Normalize test embedding
    test_embedding = test_embedding / np.linalg.norm(test_embedding)
    
    # Calculate similarity with all embeddings
    similarities = np.dot(embeddings, test_embedding.T).flatten()
    
    # Get top candidates
    top_n = 40  # Consider more candidates
    top_indices = similarities.argsort()[-top_n:][::-1]
    
    # Extract content words from test prompt
    test_content_words = extract_content_words(test_prompt)
    test_length = len(str(test_prompt).split())
    
    # Score candidates on additional metrics
    candidate_scores = []
    
    for idx in top_indices:
        train_prompt = combined_df.iloc[idx]["user_prompt"]
        train_id = combined_df.iloc[idx]["conversation_id"]
        
        # 1. Embedding similarity (from LSA)
        embedding_score = similarities[idx]
        
        # 2. Content word overlap
        train_content_words = extract_content_words(train_prompt)
        word_overlap = len(test_content_words.intersection(train_content_words)) / max(1, len(test_content_words))
        
        # 3. Length similarity
        train_length = len(str(train_prompt).split())
        length_ratio = min(test_length, train_length) / max(test_length, train_length) if max(test_length, train_length) > 0 else 0
        
        # 4. Response quality
        response_score = 0.5  # Default
        if responses_df is not None:
            response_rows = responses_df[responses_df["conversation_id"] == train_id]
            if not response_rows.empty:
                response = str(response_rows.iloc[0]["model_response"])
                
                # Response length (moderate length responses tend to have better BLEU)
                resp_len = len(response.split())
                length_score = 0.0
                if resp_len > 20 and resp_len < 300:  # Sweet spot for response length
                    length_score = 0.8
                elif resp_len <= 20:
                    length_score = 0.3
                else:
                    length_score = 0.5
                
                # Response specificity
                content_word_count = len(set(response.lower().split()).intersection(test_content_words))
                specificity_score = min(1.0, content_word_count / max(1, len(test_content_words)))
                
                # Not too generic
                generic_phrases = ['i dont know', 'cannot', 'sorry', 'ai', 'language model']
                generic_score = 1.0
                for phrase in generic_phrases:
                    if phrase in response.lower():
                        generic_score *= 0.8
                
                response_score = (length_score + specificity_score + generic_score) / 3
        
        # Calculate final combined score
        combined_score = (
            0.45 * embedding_score +  # Embedding similarity
            0.30 * word_overlap +     # Content overlap
            0.10 * length_ratio +     # Length similarity
            0.15 * response_score     # Response quality
        )
        
        candidate_scores.append((idx, combined_score))
    
    # Sort by combined score and select the best
    candidate_scores.sort(key=lambda x: x[1], reverse=True)
    best_idx = candidate_scores[0][0]
    
    return best_idx

def main(dataset="test"):
    """Main retrieval function for Track 2."""
    print(f"Running Track 2 retrieval for {dataset} dataset...")
    
    # 1) Load data
    combined_df, target_df, output_file, responses_df = load_data(dataset)
    print(f"Loaded {len(combined_df)} prompts in retrieval pool")
    print(f"Loaded {len(target_df)} {dataset} prompts")
    
    # 2) Build distributed embeddings using LSA
    embedding_data = build_embeddings(combined_df["user_prompt"])
    
    # 3) Find best matches for each target prompt
    results = []
    
    for i, row in target_df.iterrows():
        target_id = row["conversation_id"]
        test_prompt = row["user_prompt"]
        
        # Find best match
        best_idx = find_best_match(test_prompt, embedding_data, combined_df, responses_df)
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