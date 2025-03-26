#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import random
import argparse
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
import re
from tabulate import tabulate

def load_retrieval_file(filepath):
    """Load a retrieval CSV file."""
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} retrieval results from {filepath}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found - {filepath}")
        return None

def get_retrieval_files(dump_dir):
    """Get all dev retrieval files from the dump directory."""
    retrieval_files = []
    
    for filename in os.listdir(dump_dir):
        if filename.endswith('_dev.csv'):
            retrieval_files.append(os.path.join(dump_dir, filename))
    
    return retrieval_files

def preprocess_text(text):
    """Basic text preprocessing for display."""
    if not isinstance(text, str):
        return str(text)
    # Truncate long texts
    if len(text) > 100:
        return text[:97] + "..."
    return text

def calculate_bleu(reference, candidate):
    """Calculate BLEU score between reference and candidate response."""
    if not isinstance(reference, str) or not isinstance(candidate, str):
        return 0.0
    
    try:
        smoothie = SmoothingFunction().method3
        return sentence_bleu(
            [reference.split()], 
            candidate.split(), 
            weights=(0.5, 0.5, 0, 0), 
            smoothing_function=smoothie
        )
    except:
        return 0.0

def compare_retrievals(data_dir, dump_dir, num_samples=5, file_pattern=None):
    """Compare random samples from retrieval files with corresponding model responses."""
    # Load dev prompts and responses
    dev_prompts_path = os.path.join(data_dir, "dev_prompts.csv")
    dev_responses_path = os.path.join(data_dir, "dev_responses.csv")
    
    try:
        dev_prompts = pd.read_csv(dev_prompts_path)
        dev_responses = pd.read_csv(dev_responses_path)
        # Merge to get prompts and responses together
        dev_data = pd.merge(dev_prompts, dev_responses, on="conversation_id")
        print(f"Loaded {len(dev_data)} dev prompts and responses")
    except FileNotFoundError:
        print(f"Error: Could not load dev data from {data_dir}")
        return
    
    # Load train prompts and responses for lookup
    train_prompts_path = os.path.join(data_dir, "train_prompts.csv")
    train_responses_path = os.path.join(data_dir, "train_responses.csv")
    
    try:
        train_prompts = pd.read_csv(train_prompts_path)
        train_responses = pd.read_csv(train_responses_path)
        # Merge to get prompts and responses together
        train_data = pd.merge(train_prompts, train_responses, on="conversation_id")
        print(f"Loaded {len(train_data)} train prompts and responses")
    except FileNotFoundError:
        print(f"Error: Could not load train data from {data_dir}")
        return
    
    # Get all retrieval files
    retrieval_files = get_retrieval_files(dump_dir)
    if file_pattern:
        retrieval_files = [f for f in retrieval_files if file_pattern in f]
    
    if not retrieval_files:
        print(f"No retrieval files found in {dump_dir} matching pattern {file_pattern}")
        return
    
    # Select random samples from dev data
    if num_samples > len(dev_data):
        num_samples = len(dev_data)
    
    sample_indices = random.sample(range(len(dev_data)), num_samples)
    sample_ids = [dev_data.iloc[i]["conversation_id"] for i in sample_indices]
    
    # Create a lookup dictionary for train data
    train_lookup = {row["conversation_id"]: {
        "prompt": row["user_prompt"],
        "response": row["model_response"]
    } for _, row in train_data.iterrows()}
    
    # Create a lookup dictionary for dev data
    dev_lookup = {row["conversation_id"]: {
        "prompt": row["user_prompt"],
        "response": row["model_response"]
    } for _, row in dev_data.iterrows()}
    
    # Compare each retrieval file
    for file_path in retrieval_files:
        file_name = os.path.basename(file_path)
        print(f"\n========== Analyzing {file_name} ==========")
        
        retrieval_df = load_retrieval_file(file_path)
        if retrieval_df is None:
            continue
        
        # Calculate overall BLEU score
        all_bleu_scores = []
        for _, row in retrieval_df.iterrows():
            query_id = row["conversation_id"]
            retrieved_id = row["response_id"]
            
            if query_id in dev_lookup and retrieved_id in train_lookup:
                reference = dev_lookup[query_id]["response"]
                candidate = train_lookup[retrieved_id]["response"]
                bleu = calculate_bleu(reference, candidate)
                all_bleu_scores.append(bleu)
        
        average_bleu = np.mean(all_bleu_scores) if all_bleu_scores else 0
        print(f"Average BLEU score: {average_bleu:.4f}")
        
        # Create comparison table for the random samples
        table_data = []
        for sample_id in sample_ids:
            # Get the original dev prompt and response
            dev_prompt = dev_lookup[sample_id]["prompt"]
            dev_response = dev_lookup[sample_id]["response"]
            
            # Get the retrieved id and corresponding train response
            retrieval_row = retrieval_df[retrieval_df["conversation_id"] == sample_id]
            if retrieval_row.empty:
                continue
                
            retrieved_id = retrieval_row.iloc[0]["response_id"]
            train_prompt = train_lookup.get(retrieved_id, {}).get("prompt", "Not found")
            train_response = train_lookup.get(retrieved_id, {}).get("response", "Not found")
            
            # Calculate BLEU score
            bleu = calculate_bleu(dev_response, train_response)
            
            table_data.append([
                preprocess_text(dev_prompt),
                preprocess_text(train_prompt),
                preprocess_text(dev_response),
                preprocess_text(train_response),
                f"{bleu:.4f}"
            ])
        
        # Display the comparison table
        headers = ["Dev Prompt", "Retrieved Prompt", "Dev Response", "Retrieved Response", "BLEU"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))

def main():
    parser = argparse.ArgumentParser(description="Compare retrieval results from different methods")
    parser.add_argument("--data_dir", default="../data", help="Directory containing the dataset")
    parser.add_argument("--dump_dir", default="../dump", help="Directory containing the retrieval results")
    parser.add_argument("--samples", type=int, default=5, help="Number of random samples to compare")
    parser.add_argument("--pattern", help="File name pattern to filter retrieval files")
    
    args = parser.parse_args()
    
    # Get absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(script_dir, args.data_dir))
    dump_dir = os.path.abspath(os.path.join(script_dir, args.dump_dir))
    
    compare_retrievals(data_dir, dump_dir, args.samples, args.pattern)

if __name__ == "__main__":
    main()