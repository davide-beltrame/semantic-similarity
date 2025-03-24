#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# ===================== PARAMETERS =====================
DATA_DIR = "data"  # Directory for CSV data files
DUMP_DIR = "dump"  # Directory where retrieval files are saved
DEV_RESPONSES_FILE = os.path.join(DATA_DIR, "dev_responses.csv")
TRAIN_RESPONSES_FILE = os.path.join(DATA_DIR, "train_responses.csv")
# ======================================================

def run_retrieval(script_path, use_dev=True):
    """
    Runs the given retrieval script (e.g., track1_countvec.py) via a subprocess.
    The '--use_dev' flag directs the script to process the dev set.
    """
    cmd = ["python3", script_path]
    if use_dev:
        cmd.append("--use_dev")
    print("Running retrieval command:", " ".join(cmd))
    subprocess.run(cmd, check=True)

def compute_bleu_scores(retrieval_file):
    """
    Computes BLEU scores between the retrieved responses and the actual responses.
    The retrieval file should have columns: conversation_id (query) and response_id (retrieved candidate).
    Actual responses for queries are loaded from DEV_RESPONSES_FILE.
    The retrieved response text is obtained from train responses only.
    """
    # Load retrieval output (for queries)
    retrieval_df = pd.read_csv(retrieval_file)
    
    # Load actual responses for query prompts (dev set)
    dev_responses = pd.read_csv(DEV_RESPONSES_FILE)
    
    # Merge to get the actual response for each query
    merged = pd.merge(retrieval_df, dev_responses, on="conversation_id", how="left")
    merged.rename(columns={"model_response": "actual_response"}, inplace=True)
    
    # Use only train responses as the candidate pool (don't use dev responses)
    candidate_pool = pd.read_csv(TRAIN_RESPONSES_FILE)
    candidate_pool = candidate_pool[['conversation_id', 'model_response']]
    
    # Merge to get retrieved response text using the candidate's conversation_id (response_id)
    merged = pd.merge(
        merged, 
        candidate_pool, 
        left_on="response_id", 
        right_on="conversation_id",
        how="left", 
        suffixes=("", "_retrieved")
    )
    merged.rename(columns={"model_response": "retrieved_response"}, inplace=True)
    
    # Check for missing retrieved responses
    missing_count = merged['retrieved_response'].isna().sum()
    if missing_count > 0:
        print(f"Warning: {missing_count} retrieved responses could not be found in the train set.")
    
    # Check for self-matches
    self_matches = (merged['conversation_id'] == merged['response_id']).sum()
    if self_matches > 0:
        print(f"Warning: {self_matches} self-matches found. Algorithm might be retrieving dev responses.")
    
    # Compute BLEU scores for each query (using weights for 1-gram and 2-gram overlap)
    smoothing = SmoothingFunction().method3
    merged['bleu_score'] = merged.apply(
        lambda x: sentence_bleu(
            [str(x['actual_response']).split()],
            str(x['retrieved_response']).split(),
            weights=(0.5, 0.5, 0, 0),
            smoothing_function=smoothing
        ) if pd.notna(x['retrieved_response']) else 0,
        axis=1
    )
    
    avg_bleu = merged['bleu_score'].mean()
    return merged, avg_bleu

def main():
    parser = argparse.ArgumentParser(description="Tester for retrieval system")
    parser.add_argument("script", help="Path to the retrieval script (e.g., code/track1_countvec.py)")
    parser.add_argument("--output", default=None, help="Custom output file name for retrieval results")
    args = parser.parse_args()
    
    # Run retrieval on dev set
    run_retrieval(args.script, use_dev=True)
    
    # Get retrieval file name
    if args.output:
        retrieval_file = os.path.join(DUMP_DIR, args.output)
    else:
        # Try to find the file based on script name
        script_base = os.path.basename(args.script).replace('.py', '')
        potential_files = [
            os.path.join(DUMP_DIR, f"{script_base}_dev.csv"),
            os.path.join(DUMP_DIR, f"track1_{script_base}_dev.csv"),
            os.path.join(DUMP_DIR, f"track1_dev.csv")
        ]
        
        retrieval_file = None
        for file in potential_files:
            if os.path.exists(file):
                retrieval_file = file
                break
                
        if retrieval_file is None:
            print(f"Error: Retrieval file not found in {DUMP_DIR}.")
            print(f"Checked: {', '.join(potential_files)}")
            print("Please specify output file with --output.")
            sys.exit(1)
    
    print(f"Evaluating retrieval file: {retrieval_file}")
    merged, avg_bleu = compute_bleu_scores(retrieval_file)
    
    print("\nEvaluation Results:")
    print("Average BLEU score on dev set:", avg_bleu)
    
    # Save detailed results
    output_eval_file = os.path.join(DUMP_DIR, f"eval_{os.path.basename(retrieval_file)}")
    merged.to_csv(output_eval_file, index=False)
    print(f"Detailed evaluation saved to: {output_eval_file}")

if __name__ == "__main__":
    main()