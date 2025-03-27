#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
import re
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# ===================== PARAMETERS =====================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "..", "data")
DUMP_DIR = os.path.join(CURRENT_DIR, "..", "dump")
os.makedirs(DUMP_DIR, exist_ok=True)

# CSV file names
TRAIN_PROMPTS_FILE = os.path.join(DATA_DIR, "train_prompts.csv")
TRAIN_RESPONSES_FILE = os.path.join(DATA_DIR, "train_responses.csv")
DEV_PROMPTS_FILE = os.path.join(DATA_DIR, "dev_prompts.csv")
DEV_RESPONSES_FILE = os.path.join(DATA_DIR, "dev_responses.csv")
TEST_PROMPTS_FILE = os.path.join(DATA_DIR, "test_prompts.csv")

# Response filtering
MIN_RESPONSE_LENGTH = 3  # Min number of characters for valid responses

# Model parameters
#MODEL_NAME = "all-MiniLM-L6-v2" # 0.10233936335610462
MODEL_NAME = "all-mpnet-base-v2" # 0.10786308257756748
#MODEL_NAME = "paraphrase-mpnet-base-v2" # 0.10241946736959553
#MODEL_NAME = "all-distilroberta-v1" # 0.1028827366145876
#MODEL_NAME = "multi-qa-mpnet-base-dot-v1" # 0.10346886872926038
#MODEL_NAME = "gtr-t5-large" # 0.10658
#MODEL_NAME = "gtr-t5-xl" # 0.10691
#MODEL_NAME = "multi-qa-mpnet-base-dot-v1" # 0.10347
# ======================================================

def preprocess_text(text):
    """
    Enhanced preprocessing for character n-grams:
    - Lowercase, strip, normalize whitespace.
    - Add leading/trailing spaces to capture word boundaries.
    """
    if pd.isna(text):
        return ""
    text = str(text).lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return ' ' + text + ' '

def filter_invalid_responses(train_prompts, train_responses):
    """
    Filter out training examples where the response is too short or invalid.
    Returns filtered dataframes.
    """
    print(f"Filtering invalid responses (min length: {MIN_RESPONSE_LENGTH} chars)...")
    
    # Get valid response IDs
    valid_response_ids = []
    invalid_count = 0
    
    for _, row in train_responses.iterrows():
        response = row['model_response']
        conv_id = row['conversation_id']
        
        if pd.isna(response) or len(str(response).strip()) < MIN_RESPONSE_LENGTH:
            invalid_count += 1
        else:
            valid_response_ids.append(conv_id)
    
    # Filter prompts to only include those with valid responses
    filtered_train_prompts = train_prompts[train_prompts['conversation_id'].isin(valid_response_ids)].copy()
    filtered_train_responses = train_responses[train_responses['conversation_id'].isin(valid_response_ids)].copy()
    
    print(f"Filtered out {invalid_count} invalid responses. Remaining: {len(filtered_train_prompts)} examples")
    
    return filtered_train_prompts, filtered_train_responses

def load_data(use_dev=False, filter_responses=True):
    """
    Loads candidate and query prompts.
    - For dev evaluation (use_dev=True): use only train as candidate pool, and dev as queries.
    - Otherwise (for test submission): candidate pool = train + dev, query = test prompts.
    """
    # Load train data and apply filtering if requested
    train_prompts = pd.read_csv(TRAIN_PROMPTS_FILE)
    train_responses = pd.read_csv(TRAIN_RESPONSES_FILE)
    
    if filter_responses:
        train_prompts, train_responses = filter_invalid_responses(train_prompts, train_responses)
    
    if use_dev:
        candidate_df = train_prompts
        query_df = pd.read_csv(DEV_PROMPTS_FILE)
    else:
        dev_prompts = pd.read_csv(DEV_PROMPTS_FILE)
        dev_responses = pd.read_csv(DEV_RESPONSES_FILE)
        
        # For test submission, also filter dev prompts if needed
        if filter_responses:
            # Get valid dev response IDs
            valid_dev_ids = []
            for _, row in dev_responses.iterrows():
                response = row['model_response']
                conv_id = row['conversation_id']
                if not pd.isna(response) and len(str(response).strip()) >= MIN_RESPONSE_LENGTH:
                    valid_dev_ids.append(conv_id)
            
            # Filter dev prompts
            dev_prompts = dev_prompts[dev_prompts['conversation_id'].isin(valid_dev_ids)].copy()
            print(f"Using {len(dev_prompts)} valid dev prompts for candidate pool")
        
        candidate_df = pd.concat([train_prompts, dev_prompts], ignore_index=True)
        query_df = pd.read_csv(TEST_PROMPTS_FILE)
    
    # Apply preprocessing to user prompts
    candidate_df['processed_prompt'] = candidate_df['user_prompt'].apply(preprocess_text)
    query_df['processed_prompt'] = query_df['user_prompt'].apply(preprocess_text)
    
    return candidate_df, query_df

def encode_texts(model, texts):
    """
    Encodes a list/series of texts into embeddings.
    """
    return model.encode(texts.tolist(), show_progress_bar=True)

def retrieve_responses(candidate_df, query_df, use_dev=False):
    """
    For each query prompt, encode using Sentence-BERT and compute cosine similarity
    with candidate prompt embeddings. In dev mode, self-matches are excluded.
    Returns a DataFrame with query conversation_id and retrieved candidate's conversation_id.
    """
    print("Loading Sentence-BERT model...")
    model = SentenceTransformer(MODEL_NAME)
    
    print("Encoding candidate prompts...")
    # Use the preprocessed prompts instead of the original
    candidate_texts = candidate_df["processed_prompt"].fillna("").astype(str)
    candidate_embeddings = encode_texts(model, candidate_texts)
    
    print("Encoding query prompts...")
    # Use the preprocessed prompts instead of the original
    query_texts = query_df["processed_prompt"].fillna("").astype(str)
    query_embeddings = encode_texts(model, query_texts)
    
    print("Computing cosine similarities...")
    sims = cosine_similarity(query_embeddings, candidate_embeddings)
    
    if use_dev:
        # Exclude self-matches: for queries coming from dev, remove candidate with same conversation_id.
        candidate_ids = candidate_df["conversation_id"].tolist()
        query_ids = query_df["conversation_id"].tolist()
        for i, qid in enumerate(query_ids):
            for j, cid in enumerate(candidate_ids):
                if qid == cid:
                    sims[i, j] = -1  # force non-selection of self-match

    best_indices = np.argmax(sims, axis=1)
    retrieved_ids = candidate_df.iloc[best_indices]["conversation_id"].values
    
    output_df = pd.DataFrame({
        "conversation_id": query_df["conversation_id"],
        "response_id": retrieved_ids
    })
    return output_df, best_indices

def calculate_bleu_scores(query_df, candidate_df, best_indices, use_dev=False):
    """
    Calculate BLEU scores between predicted responses and actual responses.
    Only applicable when use_dev=True.
    """
    if not use_dev:
        print("BLEU score calculation only applicable for dev evaluation")
        return None, None
    
    print("Loading response data for BLEU calculation...")
    # Load response data
    train_responses_df = pd.read_csv(TRAIN_RESPONSES_FILE)
    dev_responses_df = pd.read_csv(DEV_RESPONSES_FILE)
    
    # Build response mappings
    train_response_map = dict(zip(train_responses_df['conversation_id'], train_responses_df['model_response']))
    dev_response_map = dict(zip(dev_responses_df['conversation_id'], dev_responses_df['model_response']))
    
    # Compute BLEU scores
    print("Computing BLEU scores...")
    smoothing = SmoothingFunction().method3
    bleu_scores = []
    
    for i, dev_conv_id in enumerate(query_df['conversation_id']):
        true_resp = str(dev_response_map.get(dev_conv_id, ""))
        retrieved_id = candidate_df.iloc[best_indices[i]]['conversation_id']
        pred_resp = str(train_response_map.get(retrieved_id, ""))
        
        if not true_resp or not pred_resp:
            bleu = 0.0
        else:
            bleu = sentence_bleu(
                [true_resp.split()],
                pred_resp.split(),
                weights=(0.5, 0.5, 0, 0),
                smoothing_function=smoothing
            )
        bleu_scores.append(bleu)
    
    avg_bleu = np.mean(bleu_scores)
    
    # Create detailed results dataframe
    results_df = pd.DataFrame({
        "dev_conversation_id": query_df['conversation_id'],
        "retrieved_train_id": candidate_df.iloc[best_indices]['conversation_id'].values,
        "true_response": [dev_response_map.get(cid, "") for cid in query_df['conversation_id']],
        "predicted_response": [train_response_map.get(candidate_df.iloc[idx]['conversation_id'], "") 
                              for idx in best_indices],
        "bleu_score": bleu_scores
    })
    
    return avg_bleu, results_df

def main():
    parser = argparse.ArgumentParser(description="Track 3: Retrieval using Sentence-BERT embeddings")
    parser.add_argument("--use_dev", action="store_true",
                        help="Use the dev set as queries (for evaluation). Otherwise, use test set.")
    parser.add_argument("--no_filter", action="store_true",
                        help="Don't filter invalid responses")
    args = parser.parse_args()
    
    filter_responses = not args.no_filter
    
    candidate_df, query_df = load_data(use_dev=args.use_dev, filter_responses=filter_responses)
    retrieval_df, best_indices = retrieve_responses(candidate_df, query_df, use_dev=args.use_dev)
    
    # For dev evaluation, calculate BLEU scores
    if args.use_dev:
        avg_bleu, bleu_results_df = calculate_bleu_scores(query_df, candidate_df, best_indices, use_dev=args.use_dev)
        suffix = "_filtered" if filter_responses else ""
        bleu_output_file = os.path.join(DUMP_DIR, f"track_3_sbert_preprocessed{suffix}_dev_bleu_evaluation.csv")
        bleu_results_df.to_csv(bleu_output_file, index=False)
        print(f"Average BLEU score: {avg_bleu:.5f}")
        print(f"Saved BLEU evaluation to {bleu_output_file}")
    
    # Save retrieval results
    suffix = "_filtered" if filter_responses else ""
    if args.use_dev:
        output_file = os.path.join(DUMP_DIR, f"track_3_preprocessed{suffix}_dev.csv")
    else:
        output_file = os.path.join(DUMP_DIR, f"track_3_preprocessed{suffix}_test.csv")
    
    retrieval_df.to_csv(output_file, index=False)
    print(f"Saved retrieval results to {output_file}")

if __name__ == "__main__":
    main()