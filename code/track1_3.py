import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import re
from collections import Counter

# Try to import nltk, but provide fallback if not available
try:
    import nltk
    from nltk.tokenize import word_tokenize
    
    # Download the required NLTK data if not already available
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading NLTK punkt tokenizer...")
        nltk.download('punkt', quiet=True)
    
    NLTK_AVAILABLE = True
except ImportError:
    print("NLTK not available, using simple tokenization")
    NLTK_AVAILABLE = False

def simple_tokenize(text):
    """Simple tokenization function that doesn't require NLTK."""
    # Convert to lowercase and remove punctuation
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    # Split on whitespace and filter out empty strings
    return [token for token in text.split() if token]

def tokenize(text):
    """Tokenize text, using NLTK if available, otherwise use simple tokenization."""
    if NLTK_AVAILABLE:
        return word_tokenize(text)
    else:
        return simple_tokenize(text)

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
    
    # Load train responses for analysis
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
    
    # Basic data analysis
    print("\n==== DATA ANALYSIS ====")
    
    # Prompt length distribution
    train_prompt_lengths = [len(p.split()) for p in combined_df["user_prompt"]]
    target_prompt_lengths = [len(p.split()) for p in target_df["user_prompt"]]
    
    print(f"Train prompt length stats: min={min(train_prompt_lengths)}, max={max(train_prompt_lengths)}, "
          f"avg={np.mean(train_prompt_lengths):.1f}, median={np.median(train_prompt_lengths)}")
    print(f"Target prompt length stats: min={min(target_prompt_lengths)}, max={max(target_prompt_lengths)}, "
          f"avg={np.mean(target_prompt_lengths):.1f}, median={np.median(target_prompt_lengths)}")
    
    # Response length analysis
    if responses_df is not None:
        response_lengths = [len(str(r).split()) for r in responses_df["model_response"]]
        print(f"Response length stats: min={min(response_lengths)}, max={max(response_lengths)}, "
              f"avg={np.mean(response_lengths):.1f}, median={np.median(response_lengths)}")
        
        # Sample a few prompt-response pairs for inspection
        print("\nSample prompt-response pairs:")
        for i in range(3):
            idx = np.random.randint(0, len(train_df))
            train_id = train_df.iloc[idx]["conversation_id"]
            prompt = train_df.iloc[idx]["user_prompt"]
            
            response_row = responses_df[responses_df["conversation_id"] == train_id]
            if not response_row.empty:
                response = response_row.iloc[0]["model_response"]
                print(f"Prompt {i+1}: {prompt}")
                print(f"Response {i+1}: {str(response)[:150]}..." if len(str(response)) > 150 else response)
                print("---")
    
    print("==== END DATA ANALYSIS ====\n")
    
    return combined_df, target_df, output_file, responses_df

def preprocess_text(text):
    """Preprocess text for better matching."""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters but keep spaces
    text = re.sub(r'[^\w\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_keywords(text, n=10):
    """Extract important keywords from text."""
    # Simple keyword extraction
    words = tokenize(preprocess_text(text))
    # Remove common stop words (could use a more comprehensive list)
    stop_words = {'the', 'a', 'an', 'and', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
                  'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 'from', 'of', 'as'}
    words = [w for w in words if w not in stop_words and len(w) > 1]
    
    # Get most frequent words
    word_freq = Counter(words)
    return [word for word, _ in word_freq.most_common(n)]

def build_representation(train_prompts):
    """
    Build a TF-IDF representation for prompts.
    """
    # Preprocess prompts
    print("Building TF-IDF representation...")
    processed_prompts = [preprocess_text(prompt) for prompt in train_prompts]
    
    # Build the vectorizer with improved parameters
    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),  # Include both unigrams and bigrams
        min_df=2,  # Ignore terms that appear in less than 2 documents
        max_df=0.9,  # Ignore terms that appear in more than 90% of documents
        max_features=10000  # Limit features to improve efficiency
    )
    
    train_vectors = vectorizer.fit_transform(processed_prompts)
    print(f"TF-IDF matrix shape: {train_vectors.shape}")
    
    return vectorizer, train_vectors, processed_prompts

def calculate_jaccard_similarity(set1, set2):
    """Calculate Jaccard similarity between two sets."""
    if not set1 or not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0

def get_ngrams(text, n=2):
    """Get n-grams from text without using nltk.util.ngrams."""
    tokens = tokenize(preprocess_text(text))
    ngrams_list = []
    for i in range(len(tokens) - n + 1):
        ngrams_list.append(tuple(tokens[i:i+n]))
    return set(ngrams_list)

def calculate_ngram_overlap(text1, text2, n=2):
    """Calculate n-gram overlap between two texts."""
    # Get n-grams using our custom function
    ngrams1 = get_ngrams(text1, n)
    ngrams2 = get_ngrams(text2, n)
    
    # Calculate Jaccard similarity
    return calculate_jaccard_similarity(ngrams1, ngrams2)

def find_best_match(test_prompt, vectorizer, train_vectors, combined_df, processed_prompts, responses_df=None, idx_to_debug=None):
    """
    Find most similar prompt from retrieval pool with enhanced criteria.
    """
    processed_test_prompt = preprocess_text(test_prompt)
    test_vec = vectorizer.transform([processed_test_prompt])
    
    # Calculate prompt similarity using TF-IDF
    sims = cosine_similarity(test_vec, train_vectors).flatten()
    
    # Get top matches by TF-IDF similarity
    top_n = 15
    top_indices = sims.argsort()[-top_n:][::-1]
    top_sims = sims[top_indices]
    
    # Calculate combined scores with additional criteria
    combined_scores = []
    
    for i, idx in enumerate(top_indices):
        train_prompt = combined_df.iloc[idx]["user_prompt"]
        train_id = combined_df.iloc[idx]["conversation_id"]
        
        # Initial score from TF-IDF similarity
        score = top_sims[i]
        
        # Additional criteria:
        
        # 1. Length similarity (penalize large differences in prompt length)
        test_len = len(test_prompt.split())
        train_len = len(train_prompt.split())
        len_ratio = min(test_len, train_len) / max(test_len, train_len) if max(test_len, train_len) > 0 else 0
        
        # 2. Word overlap (simpler than n-gram overlap)
        test_words = set(tokenize(processed_test_prompt))
        train_words = set(tokenize(preprocess_text(train_prompt)))
        word_overlap = calculate_jaccard_similarity(test_words, train_words)
        
        # 3. Keyword overlap
        test_keywords = set(extract_keywords(test_prompt))
        train_keywords = set(extract_keywords(train_prompt))
        keyword_overlap = calculate_jaccard_similarity(test_keywords, train_keywords)
        
        # 4. Response quality (if we have response data)
        response_quality = 0.5  # Default value
        
        # Get the response for this training example
        if responses_df is not None:
            response_rows = responses_df[responses_df["conversation_id"] == train_id]
            if not response_rows.empty:
                response = response_rows.iloc[0]["model_response"]
                # Response length as a quality indicator (normalized)
                resp_len = len(str(response).split())
                response_quality = min(1.0, resp_len / 200)  # Cap at 200 words
        
        # Combine all scores with different weights
        combined_score = (
            0.40 * score +              # TF-IDF similarity (base)
            0.20 * len_ratio +          # Length similarity
            0.20 * word_overlap +       # Word overlap (replacing bigram overlap)
            0.10 * keyword_overlap +    # Keyword overlap
            0.10 * response_quality     # Response quality
        )
        
        combined_scores.append((idx, combined_score))
    
    # Sort by combined score
    combined_scores.sort(key=lambda x: x[1], reverse=True)
    best_idx = combined_scores[0][0]
    
    # Debug information for specific samples
    if idx_to_debug is not None or np.random.random() < 0.005:  # Show 0.5% of samples
        if idx_to_debug is None:
            # Select a random sample for debugging
            idx_to_debug = np.random.randint(0, len(test_prompt))
        
        print("\n==== SIMILARITY DEBUG ====")
        print(f"Test prompt: {test_prompt}")
        
        print("\nTop 3 matches with score breakdown:")
        for i in range(min(3, len(combined_scores))):
            idx, score = combined_scores[i]
            train_prompt = combined_df.iloc[idx]["user_prompt"]
            train_id = combined_df.iloc[idx]["conversation_id"]
            
            # Calculate individual components for visualization
            tf_idf = sims[idx]
            
            test_len = len(test_prompt.split())
            train_len = len(train_prompt.split())
            len_ratio = min(test_len, train_len) / max(test_len, train_len) if max(test_len, train_len) > 0 else 0
            
            test_words = set(tokenize(processed_test_prompt))
            train_words = set(tokenize(preprocess_text(train_prompt)))
            word_overlap = calculate_jaccard_similarity(test_words, train_words)
            
            test_keywords = set(extract_keywords(test_prompt))
            train_keywords = set(extract_keywords(train_prompt))
            keyword_overlap = calculate_jaccard_similarity(test_keywords, train_keywords)
            
            print(f"{i+1}. Combined Score: {score:.4f}")
            print(f"   - TF-IDF: {tf_idf:.4f} (weight: 0.40)")
            print(f"   - Length Ratio: {len_ratio:.4f} (weight: 0.20)")
            print(f"   - Word Overlap: {word_overlap:.4f} (weight: 0.20)")
            print(f"   - Keyword Overlap: {keyword_overlap:.4f} (weight: 0.10)")
            
            print(f"   Matched Prompt: {train_prompt}")
            
            if responses_df is not None:
                response_rows = responses_df[responses_df["conversation_id"] == train_id]
                if not response_rows.empty:
                    response = response_rows.iloc[0]["model_response"]
                    resp_len = len(str(response).split())
                    response_quality = min(1.0, resp_len / 200)
                    print(f"   - Response Quality: {response_quality:.4f} (weight: 0.10)")
                    print(f"   Response: {str(response)[:150]}..." if len(str(response)) > 150 else response)
            
            print("")
        
        print("==== END SIMILARITY DEBUG ====\n")
    
    return best_idx

def main(dataset="test", debug_indices=None):
    print(f"Running retrieval for {dataset} dataset...")
    
    # 1) Load data with basic analysis
    combined_df, target_df, output_file, responses_df = load_data(dataset)
    print(f"Loaded {len(combined_df)} prompts in retrieval pool")
    print(f"Loaded {len(target_df)} {dataset} prompts")
    
    # 2) Build representation
    vectorizer, train_vectors, processed_prompts = build_representation(combined_df["user_prompt"])
    
    # 3) For each prompt, find best match
    results = []
    
    # If debug_indices is provided, these are specific indices to debug
    if debug_indices is None:
        debug_indices = []
    elif isinstance(debug_indices, int):
        debug_indices = [debug_indices]
    
    # Sample a few random indices for debugging if none provided
    if not debug_indices and np.random.random() < 0.5:  # 50% chance to add random debug indices
        debug_indices = np.random.choice(len(target_df), min(3, len(target_df)), replace=False).tolist()
    
    for i, row in target_df.iterrows():
        target_id = row["conversation_id"]
        test_prompt = row["user_prompt"]
        
        # Determine if we should debug this index
        idx_to_debug = i if i in debug_indices else None
        
        # Find best match with debug information
        best_idx = find_best_match(
            test_prompt, 
            vectorizer, 
            train_vectors, 
            combined_df, 
            processed_prompts,
            responses_df,
            idx_to_debug
        )
        
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
    
    # Print statistics
    if dataset == "dev":
        matches = sum(results_df["conversation_id"] == results_df["response_id"])
        print(f"Self-matches: {matches} out of {len(results_df)} ({matches/len(results_df)*100:.2f}%)")
    
    return output_file

if __name__ == "__main__":
    import sys
    
    # Get dataset and optional debug indices
    dataset = "dev" if len(sys.argv) <= 1 else sys.argv[1]
    
    # Optional debug indices can be provided as comma-separated values
    debug_indices = None
    if len(sys.argv) > 2:
        try:
            debug_indices = [int(idx) for idx in sys.argv[2].split(",")]
            print(f"Will debug indices: {debug_indices}")
        except ValueError:
            print("Invalid debug indices, ignoring")
    
    main(dataset, debug_indices)