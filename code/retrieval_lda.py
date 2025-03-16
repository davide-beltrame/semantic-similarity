import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import LatentDirichletAllocation
import os
import re
import string
from collections import Counter
from difflib import SequenceMatcher

# Define a simple stemmer class
class SimpleStemmer:
    def stem(self, word):
        """Very basic stemming: just remove common suffixes"""
        if len(word) < 4:
            return word
        if word.endswith('ing'):
            return word[:-3]
        if word.endswith('ed'):
            return word[:-2]
        if word.endswith('es'):
            return word[:-2]
        if word.endswith('s'):
            return word[:-1]
        return word

# Common stop words
STOP_WORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what', 
    'when', 'where', 'how', 'which', 'this', 'that', 'these', 'those', 
    'then', 'than', 'so', 'for', 'with', 'by', 'at', 'from', 'to', 'of', 'in',
    'is', 'are', 'was', 'were', 'be', 'been', 'being', 'am', 'have', 'has',
    'had', 'do', 'does', 'did', 'can', 'could', 'would', 'should', 'will',
    'shall', 'may', 'might', 'must', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
}

# Question words to give higher weight
QUESTION_WORDS = {
    'what', 'who', 'when', 'where', 'why', 'how', 'which', 'whose', 'whom', 
    'is', 'are', 'do', 'does', 'can', 'could', 'would', 'should', 'will'
}

# Initialize stemmer
stemmer = SimpleStemmer()

def preprocess_text(text, stem=True, remove_stopwords=False):
    """Enhanced preprocessing with stemming and special token handling."""
    text = str(text).lower()
    
    # Preserve question marks and important punctuation by replacing with tokens
    text = text.replace('?', ' QUESTIONMARK ')
    text = text.replace('!', ' EXCLAMATION ')
    
    # Remove other punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Split into tokens
    tokens = text.split()
    
    # Highlight question words by repeating them (gives them more weight)
    tokens = [token + ' ' + token if token in QUESTION_WORDS else token for token in tokens]
    
    # Apply stemming
    if stem:
        tokens = [stemmer.stem(token) for token in tokens]
    
    # Remove stopwords if specified
    if remove_stopwords:
        tokens = [token for token in tokens if token not in STOP_WORDS]
    
    # Rejoin tokens
    processed_text = ' '.join(tokens)
    
    # Remove extra whitespace
    processed_text = re.sub(r'\s+', ' ', processed_text).strip()
    
    return processed_text

def extract_entities(text):
    """Extract potential named entities from text."""
    # Basic approach: capitalized words might be entities
    potential_entities = []
    words = re.findall(r'\b[A-Z][a-z]*\b', str(text))
    return [word.lower() for word in words]

def calculate_jaccard_similarity(set1, set2):
    """Calculate Jaccard similarity between two sets."""
    if not set1 or not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0

def calculate_edit_distance_ratio(text1, text2):
    """Calculate normalized edit distance."""
    # Use difflib's SequenceMatcher for a Python built-in solution
    return SequenceMatcher(None, str(text1), str(text2)).ratio()

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

def build_representations(combined_df, responses_df=None):
    """Build multiple representations for prompts."""
    print("Building text representations...")
    
    # Get prompts
    prompts = combined_df["user_prompt"].tolist()
    
    # 1. TF-IDF Representation with Word N-grams
    print("Building TF-IDF word representation...")
    processed_prompts = [preprocess_text(prompt, stem=True, remove_stopwords=False) for prompt in prompts]
    
    word_vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),  # Unigrams and bigrams
        min_df=2,            # Ignore very rare terms
        max_df=0.9,          # Ignore very common terms
        max_features=10000   # Limit features for efficiency
    )
    
    word_vectors = word_vectorizer.fit_transform(processed_prompts)
    print(f"TF-IDF word matrix shape: {word_vectors.shape}")
    
    # 2. TF-IDF with Character N-grams for handling misspellings and partial matches
    print("Building TF-IDF character representation...")
    char_vectorizer = TfidfVectorizer(
        analyzer='char',
        ngram_range=(3, 5),  # Character trigrams, 4-grams, and 5-grams
        min_df=2,
        max_df=0.9,
        max_features=10000
    )
    
    char_vectors = char_vectorizer.fit_transform(processed_prompts)
    print(f"TF-IDF character matrix shape: {char_vectors.shape}")
    
    # 3. Topic Modeling with LDA
    print("Building topic model...")
    n_topics = min(50, len(prompts) // 100)  # Reasonable number of topics
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=10,  # Limit iterations for speed
        learning_method='online'
    )
    
    topic_distributions = lda.fit_transform(word_vectors)
    print(f"Topic distribution shape: {topic_distributions.shape}")
    
    # 4. Extract first words of each prompt (often contains intent)
    print("Extracting prompt characteristics...")
    first_words = []
    for prompt in prompts:
        words = str(prompt).lower().split()
        first_words.append(' '.join(words[:3]) if len(words) >= 3 else ' '.join(words))
    
    first_word_vectorizer = TfidfVectorizer(
        ngram_range=(1, 1),
        min_df=1,
        max_features=1000
    )
    first_word_vectors = first_word_vectorizer.fit_transform(first_words)
    print(f"First words matrix shape: {first_word_vectors.shape}")
    
    # 5. Pre-calculate prompt characteristics
    prompt_lengths = [len(str(p).split()) for p in prompts]
    
    # 6. Store entities for each prompt
    entities = [set(extract_entities(prompt)) for prompt in prompts]
    
    # Pack all representations into a dictionary
    representations = {
        'word_vectorizer': word_vectorizer,
        'word_vectors': word_vectors,
        'char_vectorizer': char_vectorizer,
        'char_vectors': char_vectors,
        'lda': lda,
        'topic_distributions': topic_distributions,
        'first_word_vectorizer': first_word_vectorizer,
        'first_word_vectors': first_word_vectors,
        'processed_prompts': processed_prompts,
        'prompt_lengths': prompt_lengths,
        'entities': entities,
        'prompts': prompts
    }
    
    return representations

def find_best_match(test_prompt, combined_df, representations, responses_df=None):
    """Find most similar prompt with multiple similarity metrics."""
    # Get all representations
    word_vectorizer = representations['word_vectorizer']
    word_vectors = representations['word_vectors']
    char_vectorizer = representations['char_vectorizer']
    char_vectors = representations['char_vectors']
    lda = representations['lda']
    topic_distributions = representations['topic_distributions']
    first_word_vectorizer = representations['first_word_vectorizer']
    first_word_vectors = representations['first_word_vectors']
    processed_prompts = representations['processed_prompts']
    prompt_lengths = representations['prompt_lengths']
    entities = representations['entities']
    prompts = representations['prompts']
    
    # Process test prompt
    processed_test_prompt = preprocess_text(test_prompt, stem=True, remove_stopwords=False)
    test_entities = set(extract_entities(test_prompt))
    test_length = len(str(test_prompt).split())
    
    # Get first words of test prompt
    test_words = str(test_prompt).lower().split()
    test_first_words = ' '.join(test_words[:3]) if len(test_words) >= 3 else ' '.join(test_words)
    
    # 1. TF-IDF Word similarity
    test_word_vec = word_vectorizer.transform([processed_test_prompt])
    word_sims = cosine_similarity(test_word_vec, word_vectors).flatten()
    
    # 2. TF-IDF Character similarity
    test_char_vec = char_vectorizer.transform([test_prompt.lower()])
    char_sims = cosine_similarity(test_char_vec, char_vectors).flatten()
    
    # 3. Topic similarity
    test_topic_dist = lda.transform(test_word_vec)
    topic_sims = np.zeros(len(prompts))
    for i, topic_dist in enumerate(topic_distributions):
        topic_sims[i] = np.dot(test_topic_dist[0], topic_dist)
    
    # 4. First words similarity
    test_first_word_vec = first_word_vectorizer.transform([test_first_words])
    first_word_sims = cosine_similarity(test_first_word_vec, first_word_vectors).flatten()
    
    # Get top candidates using word TF-IDF similarity
    top_n = min(20, len(prompts) // 10)  # More candidates for larger datasets
    top_indices = word_sims.argsort()[-top_n:][::-1]
    
    # Calculate combined scores for top candidates
    best_score = -1
    best_idx = top_indices[0]  # Default to highest TF-IDF
    
    for idx in top_indices:
        # Get prompt characteristics
        train_prompt = prompts[idx]
        train_length = prompt_lengths[idx]
        
        # Calculate additional similarity metrics
        
        # Length similarity
        length_ratio = min(test_length, train_length) / max(test_length, train_length) if max(test_length, train_length) > 0 else 0
        
        # Entity overlap
        train_entities = entities[idx]
        entity_overlap = calculate_jaccard_similarity(test_entities, train_entities)
        
        # Edit distance for short prompts
        edit_sim = 0
        if test_length < 10 and train_length < 10:  # Only for short prompts
            edit_sim = calculate_edit_distance_ratio(test_prompt, train_prompt)
        
        # Response quality if responses are available
        response_quality = 0.5  # Default
        train_id = combined_df.iloc[idx]["conversation_id"]
        if responses_df is not None:
            response_rows = responses_df[responses_df["conversation_id"] == train_id]
            if not response_rows.empty:
                response = response_rows.iloc[0]["model_response"]
                # Favor longer responses (up to a point)
                resp_len = len(str(response).split())
                response_quality = min(1.0, resp_len / 250)
        
        # Combine all metrics with weights
        combined_score = (
            0.35 * word_sims[idx] +        # TF-IDF word similarity
            0.15 * char_sims[idx] +        # TF-IDF character similarity
            0.15 * topic_sims[idx] +       # Topic modeling similarity
            0.10 * length_ratio +          # Length similarity
            0.10 * entity_overlap +        # Entity overlap
            0.05 * edit_sim +              # Edit distance similarity
            0.05 * first_word_sims[idx] +  # First words similarity
            0.05 * response_quality        # Response quality
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
    
    # 2) Build all representations
    representations = build_representations(combined_df, responses_df)
    
    # 3) Find best matches for each target prompt
    results = []
    
    for i, row in target_df.iterrows():
        target_id = row["conversation_id"]
        test_prompt = row["user_prompt"]
        
        # Find best match
        best_idx = find_best_match(test_prompt, combined_df, representations, responses_df)
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