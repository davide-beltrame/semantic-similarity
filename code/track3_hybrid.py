#!/usr/bin/env python3
import os
import argparse
import re
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# ===================== PARAMETERS =====================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "..", "data")
DUMP_DIR = os.path.join(CURRENT_DIR, "..", "dump")

# Create dump directory if it doesn't exist
os.makedirs(DUMP_DIR, exist_ok=True)

# CSV file names
TRAIN_PROMPTS_FILE = os.path.join(DATA_DIR, "train_prompts.csv")
TRAIN_RESPONSES_FILE = os.path.join(DATA_DIR, "train_responses.csv")
DEV_PROMPTS_FILE = os.path.join(DATA_DIR, "dev_prompts.csv")
DEV_RESPONSES_FILE = os.path.join(DATA_DIR, "dev_responses.csv")
TEST_PROMPTS_FILE = os.path.join(DATA_DIR, "test_prompts.csv")

# Model parameters
MODEL_NAME = "all-MiniLM-L6-v2"  # Pretrained Sentence-BERT model

# Hybrid approach parameters
TOP_CANDIDATES = 50  # Number of top candidates to consider for hybrid scoring
WEIGHTS = {
    'embed': 0.5,     # Embedding similarity weight
    'lexical': 0.2,   # Lexical overlap weight
    'question': 0.15, # Question type matching weight
    'length': 0.15    # Length similarity weight
}

# ======================================================

def preprocess_text(text):
    """Preprocess text to improve matching."""
    text = str(text).lower()
    
    # Replace common contractions for better matching
    replacements = {
        "don't": "do not", "doesn't": "does not", "won't": "will not",
        "can't": "cannot", "i'm": "i am", "you're": "you are",
        "it's": "it is", "that's": "that is", "there's": "there is"
    }
    for contraction, expansion in replacements.items():
        text = text.replace(contraction, expansion)
    
    # Special handling for question marks (important signal)
    has_question = '?' in text
    
    # Remove punctuation except spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text, has_question

def load_data(use_dev=False):
    """
    Loads candidate and query prompts.
    - For dev evaluation (use_dev=True): use only train as candidate pool, and dev as queries.
    - Otherwise (for test submission): candidate pool = train + dev, query = test prompts.
    """
    if use_dev:
        candidate_df = pd.read_csv(TRAIN_PROMPTS_FILE)
        query_df = pd.read_csv(DEV_PROMPTS_FILE)
        response_df = pd.read_csv(TRAIN_RESPONSES_FILE)
        try:
            query_response_df = pd.read_csv(DEV_RESPONSES_FILE)
        except:
            query_response_df = None
    else:
        train_df = pd.read_csv(TRAIN_PROMPTS_FILE)
        dev_df = pd.read_csv(DEV_PROMPTS_FILE)
        candidate_df = pd.concat([train_df, dev_df], ignore_index=True)
        query_df = pd.read_csv(TEST_PROMPTS_FILE)
        
        # Load responses
        train_responses_df = pd.read_csv(TRAIN_RESPONSES_FILE)
        try:
            dev_responses_df = pd.read_csv(DEV_RESPONSES_FILE)
            response_df = pd.concat([train_responses_df, dev_responses_df], ignore_index=True)
        except:
            response_df = train_responses_df
        
        query_response_df = None
    
    return candidate_df, query_df, response_df, query_response_df

def extract_content_words(text, stopwords=None):
    """Extract content words from text, removing stopwords."""
    if stopwords is None:
        stopwords = {
            'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what', 
            'when', 'where', 'how', 'which', 'this', 'that', 'these', 'those', 
            'then', 'to', 'of', 'in', 'for', 'with', 'by', 'at', 'from', 'is', 
            'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does',
            'did', 'can', 'could', 'will', 'would', 'should', 'i', 'you', 'he', 
            'she', 'it', 'we', 'they', 'their', 'my', 'your', 'his', 'her', 'its'
        }
    
    words = text.split()
    content_words = [word for word in words if word not in stopwords and len(word) > 2]
    return set(content_words)

def encode_texts(model, texts):
    """Encodes a list/series of texts into embeddings."""
    return model.encode(texts.tolist(), show_progress_bar=True)

def hybrid_retrieve_responses(candidate_df, query_df, response_df=None, query_response_df=None, use_dev=False):
    """
    For each query prompt, use a hybrid approach combining:
    1. Sentence-BERT embedding similarity
    2. Lexical overlap
    3. Question type matching
    4. Length similarity
    
    Returns a DataFrame with query conversation_id and retrieved candidate's conversation_id.
    """
    print("Loading Sentence-BERT model...")
    model = SentenceTransformer(MODEL_NAME)
    
    # Preprocess candidate texts
    print("Preprocessing and encoding candidate prompts...")
    candidate_texts = candidate_df["user_prompt"].fillna("").astype(str)
    candidate_processed = []
    candidate_is_question = []
    
    for text in candidate_texts:
        processed, is_question = preprocess_text(text)
        candidate_processed.append(processed)
        candidate_is_question.append(is_question)
    
    candidate_embeddings = encode_texts(model, pd.Series(candidate_processed))
    
    # Extract content words for candidates
    print("Extracting content words from candidates...")
    candidate_content_words = [extract_content_words(text) for text in candidate_processed]
    
    # Store candidate information for quick lookup
    candidate_info = {
        'texts': candidate_processed,
        'is_question': candidate_is_question,
        'lengths': [len(text.split()) for text in candidate_processed],
        'content_words': candidate_content_words,
        'ids': candidate_df["conversation_id"].tolist()
    }
    
    # Preprocess query texts
    print("Preprocessing and encoding query prompts...")
    query_texts = query_df["user_prompt"].fillna("").astype(str)
    query_processed = []
    query_is_question = []
    
    for text in query_texts:
        processed, is_question = preprocess_text(text)
        query_processed.append(processed)
        query_is_question.append(is_question)
    
    query_embeddings = encode_texts(model, pd.Series(query_processed))
    
    # Extract content words for queries
    query_content_words = [extract_content_words(text) for text in query_processed]
    query_lengths = [len(text.split()) for text in query_processed]
    
    print("Computing hybrid similarities...")
    results = []
    
    for i, query_row in enumerate(query_df.iterrows()):
        query_id = query_row[1]["conversation_id"]
        
        # Calculate embedding similarity
        embed_sims = cosine_similarity([query_embeddings[i]], candidate_embeddings)[0]
        
        # Get top candidates by embedding similarity
        top_indices = np.argsort(embed_sims)[-TOP_CANDIDATES:][::-1]
        
        # Calculate hybrid scores for top candidates
        candidate_scores = []
        
        for idx in top_indices:
            # Skip self-matches in dev mode
            if use_dev and query_id == candidate_info['ids'][idx]:
                continue
                
            # Start with embedding similarity score
            score = embed_sims[idx] * WEIGHTS['embed']
            
            # Add lexical overlap component
            if len(query_content_words[i]) > 0:
                overlap = len(query_content_words[i].intersection(candidate_info['content_words'][idx])) / len(query_content_words[i])
                score += overlap * WEIGHTS['lexical']
            
            # Add question type matching
            if query_is_question[i] == candidate_info['is_question'][idx]:
                score += WEIGHTS['question']
            
            # Add length similarity
            len_ratio = min(query_lengths[i], candidate_info['lengths'][idx]) / max(query_lengths[i], candidate_info['lengths'][idx]) if max(query_lengths[i], candidate_info['lengths'][idx]) > 0 else 0
            score += len_ratio * WEIGHTS['length']
            
            candidate_scores.append((idx, score))
        
        # Sort by score and get the best match
        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        if candidate_scores:
            best_idx = candidate_scores[0][0]
            best_id = candidate_info['ids'][best_idx]
        else:
            # Fallback to the best embedding match if no candidates after filtering
            best_idx = np.argmax(embed_sims)
            best_id = candidate_info['ids'][best_idx]
        
        results.append({
            "conversation_id": query_id,
            "response_id": best_id
        })
        
        # Print progress for every 500 queries
        if (i + 1) % 500 == 0:
            print(f"Processed {i + 1}/{len(query_df)} queries")
    
    # Create the results DataFrame with just the conversation_id and response_id columns
    output_df = pd.DataFrame(results)
    
    return output_df

def main():
    parser = argparse.ArgumentParser(description="Track 3: Hybrid retrieval with multiple similarity metrics")
    parser.add_argument("--use_dev", action="store_true",
                       help="Use the dev set as queries (for evaluation). Otherwise, use test set.")
    args = parser.parse_args()
    
    candidate_df, query_df, response_df, query_response_df = load_data(use_dev=args.use_dev)
    print(f"Loaded {len(candidate_df)} candidates and {len(query_df)} queries")
    
    retrieval_df = hybrid_retrieve_responses(
        candidate_df, query_df, response_df, query_response_df, use_dev=args.use_dev
    )
    
    if args.use_dev:
        output_file = os.path.join(DUMP_DIR, "track3_hybrid_evaluation.csv")
    else:
        output_file = os.path.join(DUMP_DIR, "track_3_test.csv")
    
    retrieval_df.to_csv(output_file, index=False)
    print(f"Saved retrieval results to {output_file}")

if __name__ == "__main__":
    main()