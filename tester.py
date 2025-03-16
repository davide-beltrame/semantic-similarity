import os
import sys
import time
import argparse
import subprocess

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
    
    # Get track from module name
    track = '1'  # Default 
    if args.module_name.startswith('track') and any(c.isdigit() for c in args.module_name):
        track = next(c for c in args.module_name if c.isdigit())
    
    # Check if module exists
    module_path = os.path.join(script_dir, "code", f"{args.module_name}.py")
    if not os.path.exists(module_path):
        print(f"Error: Module {args.module_name}.py not found!")
        return
    
    # Step 1: Run retrieval
    retrieval_start = time.time()
    
    if args.retrieval_only:
        # Import and run only retrieval
        sys.path.insert(0, os.path.join(script_dir, "code"))
        import importlib
        try:
            retrieval_module = importlib.import_module(args.module_name)
            retrieval_module.main(args.dataset)
            retrieval_time = time.time() - retrieval_start
            print(f"Retrieval completed in {retrieval_time:.2f} seconds")
        except Exception as e:
            print(f"Error running retrieval: {e}")
            return
    else:
        # Run full evaluation with evaluation_tester.py
        evaluate_path = os.path.join(script_dir, "code", "evaluate_tester.py")
        if not os.path.exists(evaluate_path):
            print(f"Error: evaluate_tester.py not found at {evaluate_path}")
            return
        
        try:
            cmd = [sys.executable, evaluate_path, args.module_name, args.dataset]
            subprocess.run(cmd, check=True)
            retrieval_time = 0  # Can't track separately when running full evaluation
        except subprocess.CalledProcessError as e:
            print(f"Error running evaluation: {e}")
            return
    
    # Print summary
    total_time = time.time() - total_start_time
    print(f"\n{'='*20} TESTING SUMMARY {'='*20}")
    print(f"Module: {args.module_name}")
    print(f"Track: {track}")
    print(f"Dataset: {args.dataset}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"{'='*20} TEST COMPLETED {'='*20}\n")

if __name__ == "__main__":
    main()