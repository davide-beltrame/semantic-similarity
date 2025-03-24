import os
import sys
import time
import argparse
import importlib
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def compute_bleu(reference, hypothesis):
    smoothing = SmoothingFunction()
    # Convert values to strings to avoid errors when calling .split()
    ref_str = str(reference) if reference is not None else ""
    hyp_str = str(hypothesis) if hypothesis is not None else ""
    return sentence_bleu([ref_str.split()], hyp_str.split(),
                         weights=(0.5, 0.5, 0, 0),
                         smoothing_function=smoothing.method3)


def main():
    parser = argparse.ArgumentParser(description='Test retrieval and evaluation for semantic similarity task.')
    parser.add_argument('module_name', type=str, help='Retrieval module name (e.g., "track2_simple")')
    parser.add_argument('--dataset', type=str, choices=['dev', 'test'], default='dev', help='Dataset to use (dev or test)')
    parser.add_argument('--retrieval-only', action='store_true', help='Run only retrieval, skip evaluation')
    
    args = parser.parse_args()
    
    print(f"\n{'='*20} TESTING {args.module_name} ON {args.dataset.upper()} {'='*20}\n")
    
    total_start_time = time.time()
    
    # Determine script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Get track number from module name (default to 1)
    track = '1'
    if args.module_name.startswith('track') and any(c.isdigit() for c in args.module_name):
        track = next(c for c in args.module_name if c.isdigit())
    
    # Check if the retrieval module exists in the code directory
    module_path = os.path.join(script_dir, "code", f"{args.module_name}.py")
    if not os.path.exists(module_path):
        print(f"Error: Module {args.module_name}.py not found!")
        return
    
    # Run retrieval: insert code directory and import the module
    retrieval_start = time.time()
    sys.path.insert(0, os.path.join(script_dir, "code"))
    try:
        retrieval_module = importlib.import_module(args.module_name)
        retrieval_module.main(args.dataset)
        retrieval_time = time.time() - retrieval_start
        print(f"Retrieval completed in {retrieval_time:.2f} seconds")
    except Exception as e:
        print(f"Error running retrieval: {e}")
        return

    # If running on DEV, load the evaluation file and compute BLEU scores
    if args.dataset == 'dev':
        # The retrieval module should output an evaluation CSV with at least:
        # 'conversation_id', 'user_prompt', 'model_response', 'retrieved_response'
        eval_csv_path = os.path.join(script_dir, f"dump/{args.module_name}_evaluation.csv")
        if not os.path.exists(eval_csv_path):
            print(f"Error: Evaluation results file not found at {eval_csv_path}")
            return

        df = pd.read_csv(eval_csv_path)
        if 'model_response' not in df.columns or 'retrieved_response' not in df.columns:
            print("Error: Required columns 'model_response' or 'retrieved_response' not found in evaluation file.")
            return
        
        bleu_scores = []
        for _, row in df.iterrows():
            bleu = compute_bleu(row['model_response'], row['retrieved_response'])
            bleu_scores.append(bleu)
        df['bleu_score'] = bleu_scores
        avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
        print(f"Average BLEU score: {avg_bleu:.4f}")
        # Optionally, update the CSV with the BLEU scores
        df.to_csv(eval_csv_path, index=False)
    
    # For the TEST dataset, we assume the retrieval module only outputs the required CSV.
    
    total_time = time.time() - total_start_time
    print(f"\n{'='*20} TESTING SUMMARY {'='*20}")
    print(f"Module: {args.module_name}")
    print(f"Track: {track}")
    print(f"Dataset: {args.dataset}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"{'='*20} TEST COMPLETED {'='*20}\n")

if __name__ == "__main__":
    main()
