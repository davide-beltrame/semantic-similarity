"""
A basic script to inspect retrieval results without complex dependencies
"""
import csv
import random
import sys

def read_csv(filename):
    """Read a CSV file and return a list of dictionaries"""
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)

def inspect_matches(dataset="dev", num_samples=5):
    """
    Inspect the retrieval matches by showing original prompts and retrieved responses
    """
    # Load predictions
    pred_file = f"track_1_{dataset}.csv"
    try:
        predictions = read_csv(pred_file)
    except FileNotFoundError:
        print(f"Error: {pred_file} not found. Run the retrieval script first.")
        return
    
    # Load prompts for the target dataset
    prompts = read_csv(f"data/{dataset}_prompts.csv")
    
    # Create a lookup dictionary for prompts
    prompt_dict = {p["conversation_id"]: p["user_prompt"] for p in prompts}
    
    # Load train data
    train_prompts = read_csv("data/train_prompts.csv")
    train_responses = read_csv("data/train_responses.csv")
    
    # Create dictionaries for train data
    train_prompt_dict = {p["conversation_id"]: p["user_prompt"] for p in train_prompts}
    train_response_dict = {r["conversation_id"]: r["model_response"] for r in train_responses}
    
    # If we have ground truth responses for this dataset, load them
    ground_truth_dict = {}
    if dataset == "dev":
        responses = read_csv(f"data/{dataset}_responses.csv")
        ground_truth_dict = {r["conversation_id"]: r["model_response"] for r in responses}
    
    # Create a list of results
    results = []
    for pred in predictions:
        target_id = pred["conversation_id"]
        retrieved_id = pred["response_id"]
        
        result = {
            "conversation_id_target": target_id,
            "user_prompt_target": prompt_dict.get(target_id, "PROMPT NOT FOUND"),
            "conversation_id_retrieved": retrieved_id,
            "user_prompt_retrieved": train_prompt_dict.get(retrieved_id, "PROMPT NOT FOUND"),
            "model_response_retrieved": train_response_dict.get(retrieved_id, "RESPONSE NOT FOUND"),
            "ground_truth_response": ground_truth_dict.get(target_id, "N/A")
        }
        results.append(result)
    
    # Select random samples
    if len(results) > num_samples:
        samples = random.sample(results, num_samples)
    else:
        samples = results
    
    # Print samples
    print(f"\n===== INSPECTING {len(samples)} RANDOM {dataset.upper()} SAMPLES =====")
    
    for i, row in enumerate(samples):
        print("\n" + "="*80)
        print(f"Sample {i+1} of {len(samples)}")
        print("-"*80)
        
        print(f"TARGET PROMPT (conversation_id: {row['conversation_id_target']}):")
        print(row['user_prompt_target'])
        print("-"*80)
        
        print(f"RETRIEVED PROMPT (conversation_id: {row['conversation_id_retrieved']}):")
        print(row['user_prompt_retrieved'])
        print("-"*80)
        
        print("RETRIEVED RESPONSE:")
        print(row['model_response_retrieved'])
        
        # Print ground truth response if available
        if dataset == "dev" and row['ground_truth_response'] != "N/A":
            print("-"*80)
            print("GROUND TRUTH RESPONSE:")
            print(row['ground_truth_response'])
    
    # Check for matches between target and retrieved conversation ids
    if dataset == "dev":
        # Count exact matches
        exact_matches = sum(1 for r in results if r['conversation_id_target'] == r['conversation_id_retrieved'])
        match_percentage = (exact_matches / len(results)) * 100 if results else 0
        print(f"\nSelf-matches: {exact_matches} out of {len(results)} ({match_percentage:.2f}%)")

if __name__ == "__main__":
    # Parse command line arguments
    dataset = "dev"  # default
    num_samples = 5  # default
    
    if len(sys.argv) > 1:
        dataset = sys.argv[1]
    if len(sys.argv) > 2:
        try:
            num_samples = int(sys.argv[2])
        except ValueError:
            print(f"Warning: Could not parse '{sys.argv[2]}' as a number. Using default: {num_samples}")
    
    inspect_matches(dataset, num_samples)