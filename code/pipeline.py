"""
Pipeline script to run retrieval and evaluation
"""
import os
import sys
from retrieval import main as retrieval_main
from evaluate_updated import evaluate_predictions

def run_pipeline(dataset="dev"):
    """
    Run the full pipeline: retrieval and evaluation
    """
    print(f"\n{'='*20} RUNNING PIPELINE FOR {dataset.upper()} {'='*20}\n")
    
    # Step 1: Run retrieval
    print(f"Step 1: Running retrieval for {dataset} dataset")
    pred_file = retrieval_main(dataset)
    
    # Step 2: Run evaluation
    print(f"\nStep 2: Running evaluation for {dataset} dataset")
    if dataset == "dev" or os.path.exists(os.path.join("data", "test_responses.csv")):
        evaluate_predictions(os.path.basename(pred_file), dataset)
    else:
        print(f"Skipping evaluation for {dataset} dataset: no ground truth responses available")
    
    print(f"\n{'='*20} PIPELINE COMPLETED {'='*20}\n")

if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] in ["dev", "test", "both"]:
        dataset = sys.argv[1]
    else:
        dataset = "both"
    
    if dataset == "both":
        run_pipeline("dev")
        run_pipeline("test")
    else:
        run_pipeline(dataset)