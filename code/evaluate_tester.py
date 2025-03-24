import pandas as pd
import os
import importlib.util
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def create_banner(title, width=60):
    """Create a centered banner with equal signs."""
    if len(title) + 2 > width:  # +2 for spaces on each side
        return f"{'=' * 5} {title} {'=' * 5}"
    
    remaining = width - len(title) - 2  # -2 for the spaces
    padding = remaining // 2
    
    # If odd number, add one more = to the end
    if remaining % 2 == 1:
        return f"{'=' * padding} {title} {'=' * (padding + 1)}"
    else:
        return f"{'=' * padding} {title} {'=' * padding}"

def import_module_from_file(module_name, file_path):
    """Import a module from file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def ensure_dump_folder():
    """Ensure dump folder exists."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dump_dir = os.path.join(os.path.dirname(script_dir), "dump")
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)
    return dump_dir

def evaluate_predictions(pred_file="dump/track_1_dev.csv", dataset="dev"):
    """
    Evaluate the predictions in the given file against the ground truth.
    """
    # Get the current directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(script_dir), "data")
    dump_dir = ensure_dump_folder()
    
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
        retrieval_pool[["conversation_id", "model_response"]],
        left_on="response_id",
        right_on="conversation_id",
        how="left",
        suffixes=("", "_retrieved")
    )
    
    # Rename columns for clarity
    merged.rename(columns={"model_response": "retrieved_response"}, inplace=True)
    
    # Check for missing values
    missing_retrievals = merged["retrieved_response"].isna().sum()
    if missing_retrievals > 0:
        print(f"Warning: {missing_retrievals} responses could not be retrieved")
    
    # 6) Compute BLEU scores
    smoothing = SmoothingFunction().method3
    
    try:
        # Calculate BLEU scores
        merged["bleu_score"] = merged.apply(
            lambda row: sentence_bleu(
                [str(row["gold_response"]).split()],
                str(row["retrieved_response"]).split(),
                weights=(0.5, 0.5, 0, 0),  # bigram BLEU
                smoothing_function=smoothing
            ),
            axis=1
        )
    except Exception as e:
        print(f"Error computing BLEU scores: {e}")
        return merged
    
    # 7) Print results and save
    print("\nSample results:")
    print(merged[["conversation_id", "retrieved_response", "gold_response", "bleu_score"]].head(5))
    
    avg_bleu = merged["bleu_score"].mean()
    print(f"\nAverage BLEU score: {avg_bleu:.4f}")
    
    # Save to dump folder
    output_file = f"dump/evaluation_{dataset}_{os.path.basename(pred_file)}"
    output_path = os.path.join(dump_dir, output_file)
    merged.to_csv(output_path, index=False)
    print(f"Detailed evaluation results saved to dump/{output_file}")
    
    return merged, avg_bleu

def run_full_evaluation(module_name, dataset="dev"):
    """Run retrieval and then evaluate the results."""
    header = f"RUNNING EVALUATION WITH {module_name} ON {dataset.upper()}"
    print(f"\n{create_banner(header)}\n")
    
    # Determine track number from module name
    track = '1'  # Default
    if module_name.startswith('track') and any(c.isdigit() for c in module_name):
        track = next(c for c in module_name if c.isdigit())
    
    # Step 1: Run retrieval
    script_dir = os.path.dirname(os.path.abspath(__file__))
    retrieval_path = os.path.join(script_dir, f"{module_name}.py")
    
    if not os.path.exists(retrieval_path):
        print(f"Error: Retrieval module not found at {retrieval_path}")
        return None
    
    print(f"Step 1: Running retrieval using {module_name} for {dataset} dataset")
    retrieval_module = import_module_from_file(module_name, retrieval_path)
    pred_file = retrieval_module.main(dataset)
    
    # Step 2: Run evaluation
    print(f"\nStep 2: Running evaluation for {dataset} dataset")
    if dataset == "dev" or os.path.exists(os.path.join("data", f"{dataset}_responses.csv")):
        result, bleu_score = evaluate_predictions(os.path.basename(pred_file), dataset)
    else:
        print(f"Skipping evaluation for {dataset} dataset: no ground truth responses available")
        bleu_score = None
    
    print(f"\n{create_banner('EVALUATION COMPLETED')}\n")
    
    return pred_file, bleu_score

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) >= 3:
        module_name = sys.argv[1]
        dataset = sys.argv[2]
    elif len(sys.argv) == 2:
        module_name = sys.argv[1]
        dataset = "dev"
    else:
        module_name = "retrieval"
        dataset = "dev"
    
    run_full_evaluation(module_name, dataset)