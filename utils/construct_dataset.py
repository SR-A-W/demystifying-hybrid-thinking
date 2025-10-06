#!/usr/bin/env python3
"""
Construct demo version of hybrid thinking dataset.
This script processes the original openr1_math_220k.jsonl dataset to create
a demo dataset with think and no_think modes.
"""

import json
import argparse
import os
import time
from typing import List, Dict, Any
try:
    from transformers import AutoTokenizer
    TOKENIZER_AVAILABLE = True
except ImportError:
    print("Warning: transformers not available, using simple word-based tokenization")
    TOKENIZER_AVAILABLE = False
from tqdm import tqdm

# Default system prompt constant
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant that can think step by step to solve problems."


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Construct demo hybrid thinking dataset")
    parser.add_argument(
        "--dataset_path", 
        type=str, 
        default="../../data/raw/openr1_math_220k.jsonl",
        help="Path to the original dataset file"
    )
    parser.add_argument(
        "--total_size", 
        type=int, 
        default=20000,
        help="Total number of samples to generate (default: 20000)"
    )
    parser.add_argument(
        "--think_ratio", 
        type=float, 
        default=0.5,
        help="Ratio of think samples (0-1, default: 0.5)"
    )
    parser.add_argument(
        "--think_min_tokens", 
        type=int, 
        default=2000,
        help="Minimum token length for think samples (default: 2000)"
    )
    parser.add_argument(
        "--think_max_tokens", 
        type=int, 
        default=8000,
        help="Maximum token length for think samples (default: 8000)"
    )
    parser.add_argument(
        "--no_think_max_tokens", 
        type=int, 
        default=500,
        help="Maximum token length for no_think samples (default: 500)"
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        default="../../data/demo/demo_hybrid_thinking.jsonl",
        help="Output path for the demo dataset"
    )
    parser.add_argument(
        "--allow_duplicates", 
        action="store_true",
        help="Allow think and no_think samples to use the same original data (default: False)"
    )
    parser.add_argument(
        "--system_prompt", 
        type=str, 
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt to use for all samples (default: uses the predefined constant)"
    )
    return parser.parse_args()


def load_original_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """Load the original dataset from JSONL file."""
    print(f"Loading original dataset from {dataset_path}...")
    data = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading data"):
            data.append(json.loads(line.strip()))
    print(f"Loaded {len(data)} samples from original dataset")
    return data


def create_think_sample(question: str, answer: str, system_prompt: str) -> Dict[str, Any]:
    """Create a think mode sample."""
    return {
        "system": system_prompt,
        "conversations": [
            {
                "from": "user",
                "value": f"{question}/think"
            },
            {
                "from": "assistant",
                "value": answer
            }
        ]
    }


def create_no_think_sample(question: str, solution: str, system_prompt: str) -> Dict[str, Any]:
    """Create a no_think mode sample."""
    return {
        "system": system_prompt,
        "conversations": [
            {
                "from": "user",
                "value": f"{question}/no_think"
            },
            {
                "from": "assistant",
                "value": f"<think>\n</think>{solution}"
            }
        ]
    }


def count_tokens(text: str, tokenizer=None) -> int:
    """Count tokens in text using tokenizer or simple word-based method."""
    if tokenizer and TOKENIZER_AVAILABLE:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        return len(tokens)
    else:
        # Simple word-based tokenization as fallback
        return len(text.split())


def build_think_dataset(data: List[Dict[str, Any]], target_size: int, 
                       min_tokens: int, max_tokens: int, tokenizer, system_prompt: str) -> tuple[List[Dict[str, Any]], set[int]]:
    """Build think mode dataset from original data."""
    print(f"Building think dataset with {target_size} samples...")
    think_samples = []
    used_indices = set()
    
    for idx, item in enumerate(tqdm(data, desc="Processing think samples")):
        if len(think_samples) >= target_size:
            break
            
        # Extract the second message (assistant's response) from messages
        if "messages" in item and len(item["messages"]) >= 2:
            assistant_message = item["messages"][1]
            if "content" in assistant_message:
                answer_content = assistant_message["content"]
                
                # Calculate token length
                token_length = count_tokens(answer_content, tokenizer)
                
                # Check if length is within range
                if min_tokens <= token_length <= max_tokens:
                    # Extract question from first message
                    user_message = item["messages"][0]
                    if "content" in user_message:
                        question = user_message["content"]
                        think_sample = create_think_sample(question, answer_content, system_prompt)
                        think_samples.append(think_sample)
                        used_indices.add(idx)
    
    print(f"Generated {len(think_samples)} think samples")
    return think_samples, used_indices


def build_no_think_dataset(data: List[Dict[str, Any]], target_size: int, 
                          max_tokens: int, tokenizer, system_prompt: str, used_indices: set[int] = None) -> List[Dict[str, Any]]:
    """Build no_think mode dataset from original data."""
    print(f"Building no_think dataset with {target_size} samples...")
    no_think_samples = []
    
    for idx, item in enumerate(tqdm(data, desc="Processing no_think samples")):
        if len(no_think_samples) >= target_size:
            break
            
        # Skip if this index was already used in think dataset (when avoiding duplicates)
        if used_indices is not None and idx in used_indices:
            continue
            
        # Extract solution from the item
        if "solution" in item and "messages" in item and len(item["messages"]) >= 1:
            solution = item["solution"]
            user_message = item["messages"][0]
            
            if "content" in user_message and solution:
                # Calculate token length of solution
                token_length = count_tokens(solution, tokenizer)
                
                # Check if length is within range
                if token_length <= max_tokens:
                    question = user_message["content"]
                    no_think_sample = create_no_think_sample(question, solution, system_prompt)
                    no_think_samples.append(no_think_sample)
    
    print(f"Generated {len(no_think_samples)} no_think samples")
    return no_think_samples


def save_dataset(samples: List[Dict[str, Any]], output_path: str):
    """Save the dataset to JSONL file."""
    print(f"Saving dataset to {output_path}...")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in tqdm(samples, desc="Saving samples"):
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"Dataset saved successfully to {output_path}")


def print_summary(think_samples: List[Dict[str, Any]], no_think_samples: List[Dict[str, Any]], elapsed_time: float):
    """Print summary statistics of the generated dataset."""
    total_samples = len(think_samples) + len(no_think_samples)
    think_count = len(think_samples)
    no_think_count = len(no_think_samples)
    
    print("\n" + "="*50)
    print("DATASET CONSTRUCTION SUMMARY")
    print("="*50)
    print(f"Total samples generated: {total_samples}")
    print(f"Think mode samples: {think_count}")
    print(f"No_think mode samples: {no_think_count}")
    print(f"Think ratio: {think_count/total_samples:.2%}")
    print(f"No_think ratio: {no_think_count/total_samples:.2%}")
    print(f"Total time elapsed: {elapsed_time:.2f} seconds")
    print(f"Average time per sample: {elapsed_time/total_samples:.3f} seconds")
    print("="*50)


def main():
    """Main function to construct the demo dataset."""
    args = parse_arguments()
    
    # Start timing
    start_time = time.time()
    
    # Initialize tokenizer
    tokenizer = None
    if TOKENIZER_AVAILABLE:
        print("Initializing tokenizer...")
        try:
            tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3", use_fast=True)
        except Exception as e:
            print(f"Warning: Failed to load tokenizer: {e}")
            print("Using simple word-based tokenization instead")
            tokenizer = None
    else:
        print("Using simple word-based tokenization")
    
    # Load original dataset
    data = load_original_dataset(args.dataset_path)
    
    # Calculate target sizes
    think_size = int(args.total_size * args.think_ratio)
    no_think_size = args.total_size - think_size
    
    print(f"Target sizes - Think: {think_size}, No_think: {no_think_size}")
    
    # Build think dataset
    think_samples, used_indices = build_think_dataset(
        data, think_size, args.think_min_tokens, args.think_max_tokens, tokenizer, args.system_prompt
    )
    
    # Build no_think dataset
    # Pass used_indices only if we want to avoid duplicates
    used_indices_for_no_think = used_indices if not args.allow_duplicates else None
    no_think_samples = build_no_think_dataset(
        data, no_think_size, args.no_think_max_tokens, tokenizer, args.system_prompt, used_indices_for_no_think
    )
    
    # Combine datasets (think first, then no_think)
    combined_samples = think_samples + no_think_samples
    
    # Save dataset
    save_dataset(combined_samples, args.output_path)
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    # Print summary
    print_summary(think_samples, no_think_samples, elapsed_time)


if __name__ == "__main__":
    main()
