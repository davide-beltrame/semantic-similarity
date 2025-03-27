#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
import re
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm

# ===================== PARAMETERS =====================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "..", "data")
DUMP_DIR = os.path.join(CURRENT_DIR, "..", "dump")
MODEL_DIR = os.path.join(CURRENT_DIR, "..", "models")
os.makedirs(DUMP_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# CSV file names
TRAIN_PROMPTS_FILE = os.path.join(DATA_DIR, "train_prompts.csv")
TRAIN_RESPONSES_FILE = os.path.join(DATA_DIR, "train_responses.csv")
DEV_PROMPTS_FILE = os.path.join(DATA_DIR, "dev_prompts.csv")
DEV_RESPONSES_FILE = os.path.join(DATA_DIR, "dev_responses.csv")
TEST_PROMPTS_FILE = os.path.join(DATA_DIR, "test_prompts.csv")

# Model parameters
BASE_MODEL_NAME = "all-mpnet-base-v2"  # Base model to fine-tune
FINETUNED_MODEL_PATH = os.path.join(MODEL_DIR, "finetuned-sbert")
# ======================================================

def preprocess_text(text):
    """
    Basic preprocessing for text:
    - Lowercase, strip, normalize whitespace.
    """
    if pd.isna(text):
        return ""
    text = str(text).lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def load_data(use_dev=False):
    """
    Loads candidate and query prompts.
    - For dev evaluation (use_dev=True): use only train as candidate pool, and dev as queries.
    - Otherwise (for test submission): candidate pool = train + dev, query = test prompts.
    """
    if use_dev:
        candidate_df = pd.read_csv(TRAIN_PROMPTS_FILE)
        query_df = pd.read_csv(DEV_PROMPTS_FILE)
    else:
        train_df = pd.read_csv(TRAIN_PROMPTS_FILE)
        dev_df = pd.read_csv(DEV_PROMPTS_FILE)
        candidate_df = pd.concat([train_df, dev_df], ignore_index=True)
        query_df = pd.read_csv(TEST_PROMPTS_FILE)
    
    # Apply preprocessing to user prompts
    candidate_df['processed_prompt'] = candidate_df['user_prompt'].apply(preprocess_text)
    query_df['processed_prompt'] = query_df['user_prompt'].apply(preprocess_text)
    
    return candidate_df, query_df

def create_training_examples(train_df, train_responses_df):
    """
    Create training examples for fine-tuning.
    We'll create pairs of prompts that should have similar responses.
    """
    print("Creating training examples for fine-tuning...")
    
    # Map conversation_ids to responses
    id_to_response = dict(zip(train_responses_df['conversation_id'], train_responses_df['model_response']))
    
    # Create a mapping from response to a list of prompts with that response
    response_to_prompts = {}
    for _, row in tqdm(train_df.iterrows(), total=len(train_df)):
        prompt = row['processed_prompt']
        conv_id = row['conversation_id']
        response = id_to_response.get(conv_id)
        
        if response and not pd.isna(response):
            response_text = str(response).strip()
            if response_text not in response_to_prompts:
                response_to_prompts[response_text] = []
            response_to_prompts[response_text].append(prompt)
    
    # Create training pairs from prompts with the same response
    train_examples = []
    for response, prompts in response_to_prompts.items():
        if len(prompts) > 1:  # Only if we have at least 2 prompts with the same response
            for i in range(len(prompts)):
                for j in range(i+1, len(prompts)):
                    # Both directions as positive examples with similarity 1.0
                    train_examples.append(InputExample(texts=[prompts[i], prompts[j]], label=1.0))
                    train_examples.append(InputExample(texts=[prompts[j], prompts[i]], label=1.0))
    
    print(f"Created {len(train_examples)} training examples")
    return train_examples

def finetune_model(train_examples, epochs=1):
    """
    Fine-tune the SentenceBERT model on our training examples.
    """
    print(f"Fine-tuning model for {epochs} epochs...")
    
    # Check if we already have a fine-tuned model
    if os.path.exists(FINETUNED_MODEL_PATH):
        print(f"Loading existing fine-tuned model from {FINETUNED_MODEL_PATH}")
        return SentenceTransformer(FINETUNED_MODEL_PATH)
    
    # Load the base model
    model = SentenceTransformer(BASE_MODEL_NAME)
    
    # Create DataLoader
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    
    # Define the training loss
    train_loss = losses.CosineSimilarityLoss(model)
    
    # Fine-tune the model
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=100,
        show_progress_bar=True
    )
    
    # Save the fine-tuned model
    model.save(FINETUNED_MODEL_PATH)
    print(f"Saved fine-tuned model to {FINETUNED_MODEL_PATH}")
    
    return model

def encode_texts(model, texts):
    """
    Encodes a list/series of texts into embeddings.
    """
    return model.encode(texts.tolist(), show_progress_bar=True)

def retrieve_responses(candidate_df, query_df, model, use_dev=False):
    """
    For each query prompt, encode using Sentence-BERT and compute cosine similarity
    with candidate prompt embeddings. In dev mode, self-matches are excluded.
    Returns a DataFrame with query conversation_id and retrieved candidate's conversation_id.
    """
    print("Encoding candidate prompts...")
    candidate_texts = candidate_df["processed_prompt"].fillna("").astype(str)
    candidate_embeddings = encode_texts(model, candidate_texts)
    
    print("Encoding query prompts...")
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
    parser = argparse.ArgumentParser(description="Track 3: Retrieval using fine-tuned SentenceBERT")
    parser.add_argument("--use_dev", action="store_true",
                        help="Use the dev set as queries (for evaluation). Otherwise, use test set.")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of epochs for fine-tuning (default: 1)")
    parser.add_argument("--no_finetune", action="store_true",
                        help="Skip fine-tuning and use the base model")
    args = parser.parse_args()
    
    # Load training data for fine-tuning
    train_df = pd.read_csv(TRAIN_PROMPTS_FILE)
    train_df['processed_prompt'] = train_df['user_prompt'].apply(preprocess_text)
    train_responses_df = pd.read_csv(TRAIN_RESPONSES_FILE)
    
    # Create training examples and fine-tune the model (or load pre-fine-tuned model)
    if not args.no_finetune:
        train_examples = create_training_examples(train_df, train_responses_df)
        model = finetune_model(train_examples, epochs=args.epochs)
    else:
        print(f"Using base model without fine-tuning: {BASE_MODEL_NAME}")
        model = SentenceTransformer(BASE_MODEL_NAME)
    
    # Load data for retrieval
    candidate_df, query_df = load_data(use_dev=args.use_dev)
    
    # Retrieve responses
    retrieval_df, best_indices = retrieve_responses(candidate_df, query_df, model, use_dev=args.use_dev)
    
    # For dev evaluation, calculate BLEU scores
    if args.use_dev:
        avg_bleu, bleu_results_df = calculate_bleu_scores(query_df, candidate_df, best_indices, use_dev=args.use_dev)
        bleu_output_file = os.path.join(DUMP_DIR, "track3_finetuned_dev_bleu_evaluation.csv")
        bleu_results_df.to_csv(bleu_output_file, index=False)
        print(f"Average BLEU score: {avg_bleu:.5f}")
        print(f"Saved BLEU evaluation to {bleu_output_file}")
    
    # Save retrieval results
    if args.use_dev:
        output_file = os.path.join(DUMP_DIR, "track_3_finetuned_dev.csv")
    else:
        output_file = os.path.join(DUMP_DIR, "track_3_finetuned_test.csv")
    
    retrieval_df.to_csv(output_file, index=False)
    print(f"Saved retrieval results to {output_file}")

if __name__ == "__main__":
    main()