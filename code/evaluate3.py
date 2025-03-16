import pandas as pd
import numpy as np
import os
import re
from collections import Counter

if __name__ == "__main__":
    main()

# Try to import nltk for BLEU, but provide fallback if not available
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("NLTK not available. Using simplified BLEU calculation.")

def simple_tokenize(text):
    """Simple tokenization function that doesn't require NLTK."""
    # Convert to lowercase and replace punctuation with spaces
    text = re.sub(r'[^\w\s]', ' ', str(text).lower())
    # Split on whitespace and filter out empty strings
    return [token for token in text.split() if token]

def simple_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0)):
    """
    Calculate a simplified BLEU score without requiring NLTK.
    This is a basic implementation that covers the core concepts.
    
    Parameters:
    - reference: list of tokens from reference text
    - candidate: list of tokens from candidate text
    - weights: tuple of weights for n-grams (default: 0.5 for unigrams, 0.5 for bigrams)
    
    Returns:
    - BLEU score between 0 and 1
    """
    if not candidate or not reference:
        return 0.0
    
    # Calculate n-gram precision for each n
    precisions = []
    
    # Handle brevity penalty
    bp = 1.0
    if len(candidate) < len(reference):
        bp = np.exp(1 - len(reference) / len(candidate)) if len(candidate) > 0 else 0.1
    
    # Calculate n-gram precisions
    for n in range(1, len(weights) + 1):
        if weights[n-1] == 0 or n > len(candidate):
            precisions.append(0.0)
            continue
            
        # Create n-grams for reference and candidate
        ref_ngrams = Counter()
        for i in range(len(reference) - n + 1):
            ref_ngrams[tuple(reference[i:i+n])] += 1
            
        cand_ngrams = Counter()
        for i in range(len(candidate) - n + 1):
            cand_ngrams[tuple(candidate[i:i+n])] += 1
        
        # Count matches (capped by reference count)
        matches = 0
        for ngram, count in cand_ngrams.items():
            matches += min(count, ref_ngrams.get(ngram, 0))
        
        # Calculate precision with smoothing (add 1 to both numerator and denominator)
        total_ngrams = max(1, len(candidate) - n + 1)
        precisions.append((matches + 1) / (total_ngrams + 1))
    
    # Combine with weights
    if all(p == 0 for p in precisions):
        return 0.0
    
    # Apply weights and calculate geometric mean
    weighted_sum = sum(w * np.log(p) for w, p in zip(weights, precisions) if w > 0)
    score = bp * np.exp(weighted_sum)
    
    return score

def calculate_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0)):
    """
    Calculate BLEU score using NLTK if available, otherwise use simplified version.
    """
    ref_tokens = simple_tokenize(reference)
    cand_tokens = simple_tokenize(candidate)
    
    if NLTK_AVAILABLE:
        smoothing = SmoothingFunction().method4  # Better for short segments
        try:
            return sentence_bleu([ref_tokens], cand_tokens, weights=weights, smoothing_function=smoothing)
        except Exception as e:
            print(f"NLTK BLEU calculation failed ({e}), falling back to simplified version")
            return simple_bleu(ref_tokens, cand_tokens, weights)
    else:
        return simple_bleu(ref_tokens, cand_tokens, weights)

def main():
    import sys
    
    pred_file = "track_1_dev.csv" if len(sys.argv) <= 1 else sys.argv[1]
    dataset = "dev" if len(sys.argv) <= 2 else sys.argv[2]
    debug = False if len(sys.argv) <= 3 else sys.argv[3].lower() == "debug"
    
    evaluate_predictions(pred_file, dataset, debug)

def evaluate_predictions(pred_file="track_1_dev.csv", dataset="dev", debug=False):
    """
    Evaluate the predictions in the given file against the ground truth.
    """
    # Get the current directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(script_dir), "data")
    
    # Construct file paths
    pred_path = os.path.join(os.path.dirname(script_dir), pred_file)
    
    print(f"Evaluating {pred_file} against {dataset} dataset...")
    
    # 1) Load predictions
    try:
        pred_df = pd.read_csv(pred_path)
        print(f"Loaded {len(pred_df)} predictions from {pred_file}")
    except FileNotFoundError:
        print(f"Error: {pred_file} not found. Run the retrieval script first.")
        return
    
    # 2) Load ground truth data
    prompts_path = os.path.join(data_dir, f"{dataset}_prompts.csv")
    responses_path = os.path.join(data_dir, f"{dataset}_responses.csv")
    
    try:
        prompts_df = pd.read_csv(prompts_path)
        responses_df = pd.read_csv(responses_path)
        print(f"Loaded {len(prompts_df)} prompts and {len(responses_df)} responses for {dataset} set")
    except FileNotFoundError:
        print(f"Error: Required data files for {dataset} not found.")
        return
    
    # Merge to get ground truth
    gold_df = pd.merge(prompts_df, responses_df, on="conversation_id", how="left")
    gold_df.rename(columns={"model_response": "gold_response"}, inplace=True)
    
    # 3) Load retrieval pool
    train_prompts_path = os.path.join(data_dir, "train_prompts.csv")
    train_responses_path = os.path.join(data_dir, "train_responses.csv")
    
    train_prompts = pd.read_csv(train_prompts_path)
    train_responses = pd.read_csv(train_responses_path)
    
    # Merge train prompts and responses
    train_data = pd.merge(train_prompts, train_responses, on="conversation_id", how="left")
    
    # For test, include dev data in retrieval pool
    if dataset == "test":
        dev_prompts = pd.read_csv(os.path.join(data_dir, "dev_prompts.csv"))
        dev_responses = pd.read_csv(os.path.join(data_dir, "dev_responses.csv"))
        dev_data = pd.merge(dev_prompts, dev_responses, on="conversation_id", how="left")
        retrieval_pool = pd.concat([train_data, dev_data], ignore_index=True)
    else:
        retrieval_pool = train_data.copy()
    
    print(f"Retrieval pool contains {len(retrieval_pool)} entries")
    
    # 4) Merge predictions with ground truth
    merged = pd.merge(pred_df, gold_df, on="conversation_id", how="left")
    
    # 5) Merge with retrieval pool to get retrieved responses
    merged = pd.merge(
        merged,
        retrieval_pool[["conversation_id", "model_response", "user_prompt"]],
        left_on="response_id",
        right_on="conversation_id",
        how="left",
        suffixes=("", "_retrieved")
    )
    
    # Rename columns for clarity
    merged.rename(columns={
        "model_response": "retrieved_response",
        "user_prompt": "original_prompt",
        "user_prompt_retrieved": "matched_prompt"
    }, inplace=True)
    
    # Check for missing values
    missing_retrievals = merged["retrieved_response"].isna().sum()
    if missing_retrievals > 0:
        print(f"Warning: {missing_retrievals} responses could not be retrieved")
    
    # Calculate prompt and response lengths
    merged["prompt_length"] = merged["original_prompt"].apply(lambda x: len(str(x).split()))
    merged["gold_length"] = merged["gold_response"].apply(lambda x: len(str(x).split()))
    merged["retrieved_length"] = merged["retrieved_response"].apply(lambda x: len(str(x).split()))
    
    # Try different BLEU configurations
    print("\nCalculating BLEU scores...")
    
    bleu_configs = [
        ("BLEU-1", (1.0, 0.0, 0.0, 0.0)),
        ("BLEU-2", (0.5, 0.5, 0.0, 0.0))
    ]
    
    for bleu_name, weights in bleu_configs:
        merged[f"{bleu_name}_score"] = merged.apply(
            lambda row: calculate_bleu(row["gold_response"], row["retrieved_response"], weights),
            axis=1
        )
        avg_score = merged[f"{bleu_name}_score"].mean()
        print(f"{bleu_name} average score: {avg_score:.4f}")
    
    # Use BLEU-2 as the default for backward compatibility
    merged["bleu_score"] = merged["BLEU-2_score"]
    
    # If debug mode, analyze extreme cases
    if debug:
        # Show best and worst matches
        print("\n===== EXTREME CASES ANALYSIS =====")
        
        # Worst cases
        print("\nWorst 3 matches:")
        worst_cases = merged.sort_values("bleu_score").head(3)
        for i, row in worst_cases.iterrows():
            print(f"BLEU: {row['bleu_score']:.4f}, ID: {row['conversation_id']}")
            print(f"Prompt: {row['original_prompt']}")
            print(f"Gold Response: {str(row['gold_response'])[:150]}..." if len(str(row['gold_response'])) > 150 else row['gold_response'])
            print(f"Retrieved Response: {str(row['retrieved_response'])[:150]}..." if len(str(row['retrieved_response'])) > 150 else row['retrieved_response'])
            print("---")
        
        # Best cases
        print("\nBest 3 matches:")
        best_cases = merged.sort_values("bleu_score", ascending=False).head(3)
        for i, row in best_cases.iterrows():
            print(f"BLEU: {row['bleu_score']:.4f}, ID: {row['conversation_id']}")
            print(f"Prompt: {row['original_prompt']}")
            print(f"Gold Response: {str(row['gold_response'])[:150]}..." if len(str(row['gold_response'])) > 150 else row['gold_response'])
            print(f"Retrieved Response: {str(row['retrieved_response'])[:150]}..." if len(str(row['retrieved_response'])) > 150 else row['retrieved_response'])
            print("---")
        
        print("===== END EXTREME CASES ANALYSIS =====")
    
    # Print sample results
    print("\nSample results:")
    print(merged[["conversation_id", "retrieved_response", "gold_response", "bleu_score"]].head(5))
    
    avg_bleu = merged["bleu_score"].mean()
    print(f"\nAverage BLEU score: {avg_bleu:.4f}")
    
    # Calculate correlations between BLEU and other factors
    print("\nCorrelations with BLEU score:")
    bleu_vs_prompt_len = np.corrcoef(merged["bleu_score"], merged["prompt_length"])[0, 1]
    print(f"Prompt length correlation: {bleu_vs_prompt_len:.4f}")
    
    bleu_vs_response_len = np.corrcoef(merged["bleu_score"], merged["retrieved_length"])[0, 1]
    print(f"Response length correlation: {bleu_vs_response_len:.4f}")
    
    # Save the evaluation results
    output_file = f"evaluation_{dataset}_{os.path.basename(pred_file)}"
    output_path = os.path.join(os.path.dirname(script_dir), output_file)
    merged.to_csv(output_path, index=False)
    print(f"Detailed evaluation results saved to {output_file}")
    
    return merged