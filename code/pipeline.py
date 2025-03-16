"""
Improved pipeline script to run retrieval and evaluation
with robust similarity metrics and better BLEU scoring
"""
import os
import sys
import time
import importlib.util

def check_module_exists(module_name, rename_from=None):
    """Check if a module exists at the specified path and import it."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    module_path = os.path.join(script_dir, f"{module_name}.py")
    
    if os.path.exists(module_path):
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    elif rename_from and os.path.exists(os.path.join(script_dir, f"{rename_from}.py")):
        # Try to rename the file
        try:
            os.rename(os.path.join(script_dir, f"{rename_from}.py"), module_path)
            print(f"Renamed {rename_from}.py to {module_name}.py")
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        except Exception as e:
            print(f"Error renaming {rename_from}.py to {module_name}.py: {e}")
            return None
    else:
        print(f"Warning: Module {module_name}.py not found")
        return None

def run_pipeline(dataset="dev", debug=False, sample_indices=None):
    """
    Run the full pipeline: retrieval and evaluation
    
    Parameters:
    dataset (str): 'dev' or 'test'
    debug (bool): Whether to run in debug mode with detailed analysis
    sample_indices (list): Optional specific indices to debug
    """
    start_time = time.time()
    print(f"\n{'='*20} RUNNING PIPELINE FOR {dataset.upper()} {'='*20}\n")
    
    # Dynamically load modules
    retrieval_module = check_module_exists("retrieval", "retrieval2")
    evaluate_module = check_module_exists("evaluate", "evaluate2")
    
    if not retrieval_module:
        print("Error: Retrieval module not found. Pipeline cannot continue.")
        return
    
    # Step 1: Run retrieval
    print(f"Step 1: Running retrieval for {dataset} dataset")
    
    # Pass sample indices to retrieval for debugging if provided
    retrieval_main = getattr(retrieval_module, "main")
    pred_file = retrieval_main(dataset, sample_indices)
    
    retrieval_time = time.time()
    print(f"Retrieval completed in {retrieval_time - start_time:.2f} seconds")
    
    # Step 2: Run evaluation
    print(f"\nStep 2: Running evaluation for {dataset} dataset")
    if evaluate_module and (dataset == "dev" or os.path.exists(os.path.join("data", f"{dataset}_responses.csv"))):
        evaluate_predictions = getattr(evaluate_module, "evaluate_predictions")
        evaluate_predictions(os.path.basename(pred_file), dataset, debug)
        print(f"Evaluation completed in {time.time() - retrieval_time:.2f} seconds")
    else:
        print(f"Skipping evaluation for {dataset} dataset: no ground truth responses available or evaluation module missing")
    
    total_time = time.time() - start_time
    print(f"\nTotal pipeline execution time: {total_time:.2f} seconds")
    print(f"\n{'='*20} PIPELINE COMPLETED {'='*20}\n")

if __name__ == "__main__":
    # Parse command line arguments
    args = sys.argv[1:]
    
    # Set defaults
    dataset = "both"
    debug = False
    sample_indices = None
    
    # Parse arguments
    i = 0
    while i < len(args):
        if args[i] in ["dev", "test", "both"]:
            dataset = args[i]
        elif args[i] == "--debug":
            debug = True
        elif args[i] == "--indices" and i + 1 < len(args):
            try:
                sample_indices = [int(idx) for idx in args[i+1].split(",")]
                i += 1  # Skip the next argument as we've processed it
            except ValueError:
                print("Warning: Invalid sample indices format. Should be comma-separated integers.")
        else:
            print(f"Warning: Ignoring unknown argument: {args[i]}")
        i += 1
    
    print(f"Running pipeline with dataset={dataset}, debug={debug}")
    if sample_indices:
        print(f"Will debug specific indices: {sample_indices}")
    
    if dataset == "both":
        run_pipeline("dev", debug, sample_indices)
        run_pipeline("test", debug, sample_indices)
    else:
        run_pipeline(dataset, debug, sample_indices)