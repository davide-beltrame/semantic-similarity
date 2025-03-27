#!/usr/bin/env python3
import os
import pandas as pd
import argparse
from glob import glob
from itertools import combinations

def main():
    parser = argparse.ArgumentParser(description="Compare prediction overlaps across multiple CSV files")
    parser.add_argument("--dir", type=str, default="./dump", 
                        help="Directory containing prediction CSV files (default: ./dump)")
    parser.add_argument("--pattern", type=str, default="*.csv", 
                        help="Pattern to match CSV files (default: *.csv)")
    parser.add_argument("--output", type=str, default="comparison_results.txt",
                        help="Output file for comparison results (default: comparison_results.txt)")
    parser.add_argument("--min_overlap", type=int, default=0,
                        help="Minimum number of matching predictions to include in results (default: 0)")
    args = parser.parse_args()
    
    # Find all CSV files matching the pattern in the specified directory
    csv_files = glob(os.path.join(args.dir, args.pattern))
    
    if not csv_files:
        print(f"No CSV files found in {args.dir} matching pattern {args.pattern}")
        return
    
    print(f"Found {len(csv_files)} CSV files to compare.")
    
    # Dictionary to store DataFrames
    dataframes = {}
    
    # Load all CSV files
    for file_path in csv_files:
        file_name = os.path.basename(file_path).split('.')[0]  # Get filename without extension
        try:
            df = pd.read_csv(file_path)
            # Check if the CSV has the required columns
            if "conversation_id" in df.columns and "response_id" in df.columns:
                # Set conversation_id as index for easier comparison
                df = df.set_index("conversation_id")
                dataframes[file_name] = df
            else:
                print(f"Warning: {file_path} doesn't have required columns and will be skipped.")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    if len(dataframes) < 2:
        print("Need at least two valid CSV files to compare.")
        return
    
    # Generate all possible pairs of DataFrames
    results = []
    for (name1, df1), (name2, df2) in combinations(dataframes.items(), 2):
        # Ensure both dataframes have the same indices (conversation_ids)
        common_ids = df1.index.intersection(df2.index)
        
        if len(common_ids) == 0:
            print(f"No common conversation_ids between {name1} and {name2}.")
            continue
        
        # Compare response_ids for common conversation_ids
        df1_subset = df1.loc[common_ids]
        df2_subset = df2.loc[common_ids]
        
        # Count exact matches
        matches = (df1_subset["response_id"] == df2_subset["response_id"]).sum()
        
        if matches >= args.min_overlap:
            results.append((name1, name2, matches))
    
    # Sort results by number of matches in descending order
    results.sort(key=lambda x: x[2], reverse=True)
    
    # Write results to output file
    with open(args.output, 'w') as f:
        for name1, name2, matches in results:
            line = f"{name1} & {name2} = {matches}\n"
            f.write(line)
            print(line, end='')
    
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()