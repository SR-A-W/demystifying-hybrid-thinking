#!/usr/bin/env python3
"""
Split hybrid thinking dataset into two phases.
This script splits the dataset by extracting a specified number of /think samples
from the beginning of the dataset.
"""

import json
import argparse
import os
from typing import List, Dict, Any, Tuple
from tqdm import tqdm


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Split hybrid thinking dataset into two phases")
    parser.add_argument(
        "--input_path", 
        type=str, 
        required=True,
        help="Path to the input dataset file"
    )
    parser.add_argument(
        "--split_size", 
        type=int, 
        required=True,
        help="Number of /think samples to extract for phase1"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./",
        help="Output directory for the split datasets (default: current directory)"
    )
    parser.add_argument(
        "--output_prefix", 
        type=str, 
        default="split_dataset",
        help="Prefix for output filenames (default: split_dataset)"
    )
    return parser.parse_args()


def load_dataset(input_path: str) -> List[Dict[str, Any]]:
    """Load the dataset from JSONL file."""
    print(f"Loading dataset from {input_path}...")
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading data"):
            data.append(json.loads(line.strip()))
    print(f"Loaded {len(data)} samples from dataset")
    return data


def find_split_point(data: List[Dict[str, Any]], target_think_count: int) -> Tuple[int, int, bool]:
    """
    Find the split point in the dataset.
    
    Returns:
        Tuple of (split_index, actual_think_count, is_correct_split)
        - split_index: index where to split the dataset
        - actual_think_count: actual number of /think samples found
        - is_correct_split: whether the split is correct (ends with /think)
    """
    think_count = 0
    split_index = 0
    
    for idx, sample in enumerate(data):
        # Check if this is a /think sample
        if (sample.get("conversations") and 
            len(sample["conversations"]) > 0 and 
            sample["conversations"][0].get("from") == "user" and
            sample["conversations"][0].get("value", "").endswith("/think")):
            
            think_count += 1
            
            if think_count == target_think_count:
                split_index = idx + 1
                break
    
    # If we didn't find enough /think samples, use all available /think samples
    insufficient_think = False
    if think_count < target_think_count:
        # Find the last /think sample
        for idx in range(len(data) - 1, -1, -1):
            sample = data[idx]
            if (sample.get("conversations") and 
                len(sample["conversations"]) > 0 and 
                sample["conversations"][0].get("from") == "user" and
                sample["conversations"][0].get("value", "").endswith("/think")):
                split_index = idx + 1
                break
        # If still no split point found, use all data
        if split_index == 0:
            split_index = len(data)
        insufficient_think = True
        print(f"Warning: Only found {think_count} /think samples, less than requested {target_think_count}")
    
    # Check if the split is correct (ends with /think)
    is_correct_split = True
    if split_index > 0 and split_index < len(data):
        last_sample = data[split_index - 1]
        if (last_sample.get("conversations") and 
            len(last_sample["conversations"]) > 0 and 
            last_sample["conversations"][0].get("from") == "user"):
            last_value = last_sample["conversations"][0].get("value", "")
            if not last_value.endswith("/think"):
                is_correct_split = False
    
    return split_index, think_count, is_correct_split


def save_dataset(samples: List[Dict[str, Any]], output_path: str):
    """Save the dataset to JSONL file."""
    print(f"Saving dataset to {output_path}...")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in tqdm(samples, desc="Saving samples"):
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"Dataset saved successfully to {output_path}")


def print_summary(phase1_samples: List[Dict[str, Any]], phase2_samples: List[Dict[str, Any]], 
                 target_think_count: int, actual_think_count: int, is_correct_split: bool):
    """Print summary statistics of the split dataset."""
    print("\n" + "="*50)
    print("DATASET SPLIT SUMMARY")
    print("="*50)
    print(f"Phase 1 samples: {len(phase1_samples)}")
    print(f"Phase 2 samples: {len(phase2_samples)}")
    print(f"Total samples: {len(phase1_samples) + len(phase2_samples)}")
    print(f"Target /think samples for Phase 1: {target_think_count}")
    print(f"Actual /think samples in Phase 1: {actual_think_count}")
    print(f"Split is correct (ends with /think): {'Yes' if is_correct_split else 'No'}")
    
    if not is_correct_split:
        print("WARNING: Split is incorrect! Phase 1 does not end with /think sample.")
    
    insufficient_think = actual_think_count < target_think_count
    if insufficient_think:
        print("WARNING: Insufficient /think samples! Requested more than available.")
    
    print("="*50)


def main():
    """Main function to split the dataset."""
    args = parse_arguments()
    
    # Load dataset
    data = load_dataset(args.input_path)
    
    # Find split point
    split_index, actual_think_count, is_correct_split = find_split_point(data, args.split_size)
    
    # Split the dataset
    phase1_samples = data[:split_index]
    phase2_samples = data[split_index:]
    
    # Determine output filenames
    base_name = os.path.splitext(os.path.basename(args.input_path))[0]
    output_dir = args.output_dir
    
    # Check if we have insufficient /think samples (over-split)
    insufficient_think = actual_think_count < args.split_size
    
    if is_correct_split and not insufficient_think:
        phase1_filename = f"{args.output_prefix}_phase1.jsonl"
        phase2_filename = f"{args.output_prefix}_phase2.jsonl"
    else:
        warning_suffix = ""
        if not is_correct_split:
            warning_suffix += "_WARNING_incorrect_split"
        if insufficient_think:
            warning_suffix += "_WARNING_insufficient_think"
        
        phase1_filename = f"{args.output_prefix}_phase1{warning_suffix}.jsonl"
        phase2_filename = f"{args.output_prefix}_phase2{warning_suffix}.jsonl"
    
    phase1_path = os.path.join(output_dir, phase1_filename)
    phase2_path = os.path.join(output_dir, phase2_filename)
    
    # Save datasets
    save_dataset(phase1_samples, phase1_path)
    save_dataset(phase2_samples, phase2_path)
    
    # Print summary
    print_summary(phase1_samples, phase2_samples, args.split_size, actual_think_count, is_correct_split)
    
    if not is_correct_split or insufficient_think:
        print(f"\nWARNING: Issues detected with the split!")
        print(f"Phase 1 file: {phase1_path}")
        print(f"Phase 2 file: {phase2_path}")
        if not is_correct_split:
            print("- Files marked with 'WARNING_incorrect_split' due to incorrect split")
        if insufficient_think:
            print("- Files marked with 'WARNING_insufficient_think' due to insufficient /think samples")


if __name__ == "__main__":
    main()
