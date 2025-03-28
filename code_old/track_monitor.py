#!/usr/bin/env python3
import json
import os
import time
import datetime
import argparse
from tabulate import tabulate

def load_json_safely(filepath):
    """Load JSON file with error handling"""
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
        return None
    except json.JSONDecodeError:
        # Handle case where file is being written
        print(f"Warning: Could not decode JSON in {filepath}. File might be currently being written.")
        return None
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def format_config(config, track):
    """Format config for display based on track"""
    if config is None:
        return "No data available"
    
    if track == 1:
        # Format Track 1 config (discrete representation)
        vect_type = config.get("vectorizer_type", "N/A")
        analyzer = config.get("analyzer", "N/A")
        ngram = config.get("ngram_range", "N/A")
        preprocessing = config.get("preprocessing", "standard")
        
        return f"Vectorizer: {vect_type}, Analyzer: {analyzer}, NGram: {ngram}, Preproc: {preprocessing}"
        
    elif track == 2:
        # Format Track 2 config (distributed representation)
        model_type = config.get("model_type", "N/A")
        vs = config.get("vector_size", "N/A")
        window = config.get("window", "N/A")
        emb_method = config.get("embedding_method", "N/A")
        
        return f"Model: {model_type}, Vec Size: {vs}, Window: {window}, Emb Method: {emb_method}"
        
    elif track == 3:
        # Format Track 3 config (hybrid)
        components = config.get("components", [])
        weights = config.get("weights", [])
        
        if not components:
            return "No components specified"
            
        comp_desc = []
        for i, comp in enumerate(components):
            comp_desc.append(f"{comp}={weights[i]:.2f}")
        
        return " + ".join(comp_desc)
    
    return "Unknown track format"

def monitor_tracks(results_dir, interval=10, table_format="simple"):
    """Monitor track results, refreshing at specified interval"""
    track_files = {
        1: os.path.join(results_dir, "track_1_grid_search_results.json"),
        2: os.path.join(results_dir, "track_2_grid_search_results.json"),
        3: os.path.join(results_dir, "track_3_hybrid_grid_search_results.json")
    }
    
    print(f"Monitoring track results in: {results_dir}")
    print(f"Refresh interval: {interval} seconds")
    print("Press Ctrl+C to exit")
    print("\n")
    
    try:
        while True:
            track_data = []
            max_bleu = 0
            best_track = None
            
            # Gather data from all tracks
            for track, filepath in track_files.items():
                data = load_json_safely(filepath)
                
                if data is None:
                    track_data.append([track, "N/A", "File not found or not readable", "N/A"])
                    continue
                
                best_config = data.get('best_config', {})
                best_bleu = data.get('best_bleu', 0)
                
                if best_bleu > max_bleu:
                    max_bleu = best_bleu
                    best_track = track
                
                last_modified = datetime.datetime.fromtimestamp(os.path.getmtime(filepath)).strftime('%Y-%m-%d %H:%M:%S')
                config_summary = format_config(best_config, track)
                
                track_data.append([track, f"{best_bleu:.5f}", config_summary, last_modified])
            
            # Clear screen (works on both Unix/Linux and Windows)
            os.system('cls' if os.name == 'nt' else 'clear')
            
            # Print current time
            print(f"Last updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Format data with best track highlighted
            formatted_data = []
            for row in track_data:
                track_num = row[0]
                if track_num == best_track:
                    # Highlight best track (using ANSI escape codes for terminal colors)
                    row = [f"\033[1m\033[92m{item}\033[0m" for item in row]  # Bold and green
                formatted_data.append(row)
            
            # Print table
            headers = ["Track", "Best BLEU", "Configuration", "Last Modified"]
            print(tabulate(formatted_data, headers=headers, tablefmt=table_format))
            
            # Wait for next refresh
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")

def main():
    parser = argparse.ArgumentParser(description="Monitor track results in real-time")
    parser.add_argument("--results_dir", type=str, default="../results",
                        help="Directory containing result JSON files")
    parser.add_argument("--interval", type=int, default=10,
                        help="Refresh interval in seconds")
    parser.add_argument("--format", type=str, default="grid",
                        choices=["plain", "simple", "grid", "fancy_grid", "pipe"],
                        help="Table display format")
    
    args = parser.parse_args()
    
    # Ensure results directory exists
    if not os.path.exists(args.results_dir):
        print(f"Results directory {args.results_dir} does not exist.")
        return
    
    monitor_tracks(args.results_dir, args.interval, args.format)

if __name__ == "__main__":
    main()