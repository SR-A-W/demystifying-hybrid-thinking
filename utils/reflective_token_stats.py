#!/usr/bin/env python3
"""
Reflective Token Statistics Script

This script analyzes evaluation results to count reflective tokens in model responses.
Reflective tokens are reasoning-related words that indicate thinking processes.

Usage:
    python reflective_token_stats.py --input_dir /path/to/eval/results
"""

import argparse
import json
import os
import re
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Any, Tuple


# Define reflective tokens to count (simplified list)
REFLECTIVE_TOKENS = [
    "wait", "hmm", "okay", "alternatively", "let me think"
]


def normalize_text(text: str) -> str:
    """Normalize text for token matching."""
    # Convert to lowercase and remove extra whitespace
    return re.sub(r'\s+', ' ', text.lower().strip())


def count_reflective_tokens(text: str, tokens: List[str]) -> Dict[str, int]:
    """
    Count occurrences of reflective tokens in text.
    
    Args:
        text: The text to analyze
        tokens: List of reflective tokens to count
        
    Returns:
        Dictionary mapping token to count
    """
    normalized_text = normalize_text(text)
    token_counts = defaultdict(int)
    
    for token in tokens:
        # Create pattern to match word boundaries
        pattern = r'\b' + re.escape(token.lower()) + r'\b'
        matches = re.findall(pattern, normalized_text)
        token_counts[token] = len(matches)
    
    return dict(token_counts)


def analyze_responses(responses: List[Dict[str, Any]], tokens: List[str]) -> Dict[str, Any]:
    """
    Analyze all responses in a single evaluation result.
    
    Args:
        responses: List of response dictionaries
        tokens: List of reflective tokens to count
        
    Returns:
        Dictionary containing statistics
    """
    total_responses = len(responses)
    total_tokens = defaultdict(int)
    response_token_counts = []
    
    for response in responses:
        content = response.get('content', '')
        if not content:
            continue
            
        token_counts = count_reflective_tokens(content, tokens)
        response_token_counts.append(token_counts)
        
        # Accumulate total counts
        for token, count in token_counts.items():
            total_tokens[token] += count
    
    # Calculate averages
    avg_tokens_per_response = {}
    for token in tokens:
        total_count = total_tokens[token]
        avg_tokens_per_response[token] = total_count / total_responses if total_responses > 0 else 0
    
    # Calculate total reflective tokens per response
    total_reflective_per_response = []
    for response_counts in response_token_counts:
        total_reflective = sum(response_counts.values())
        total_reflective_per_response.append(total_reflective)
    
    avg_total_reflective = sum(total_reflective_per_response) / len(total_reflective_per_response) if total_reflective_per_response else 0
    
    return {
        'total_responses': total_responses,
        'avg_total_reflective_tokens_per_response': avg_total_reflective,
        'avg_tokens_per_response': avg_tokens_per_response,
        'total_token_counts': dict(total_tokens),
        'response_breakdown': response_token_counts
    }


def process_result_file(file_path: Path, tokens: List[str], exhaustive: bool = False) -> Dict[str, Any]:
    """
    Process a single results.json file.
    
    Args:
        file_path: Path to the results.json file
        tokens: List of reflective tokens to count
        exhaustive: Whether to include detailed problem-by-problem statistics
        
    Returns:
        Dictionary containing analysis results
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Process each problem in the results
        problem_stats = {} if exhaustive else None
        overall_stats = {
            'total_problems': 0,
            'total_responses': 0,
            'avg_total_reflective_tokens_per_response': 0,
            'avg_tokens_per_response': defaultdict(float),
            'total_token_counts': defaultdict(int)
        }
        
        total_problems = len(data)
        processed_problems = 0
        
        for problem_id, problem_data in data.items():
            responses = problem_data.get('responses', [])
            if not responses:
                continue
                
            if exhaustive:
                problem_analysis = analyze_responses(responses, tokens)
                problem_stats[problem_id] = problem_analysis
                
                # Accumulate overall statistics
                overall_stats['total_problems'] += 1
                overall_stats['total_responses'] += problem_analysis['total_responses']
                
                # Accumulate token counts
                for token, count in problem_analysis['total_token_counts'].items():
                    overall_stats['total_token_counts'][token] += count
            else:
                # Simplified processing for non-exhaustive mode
                overall_stats['total_problems'] += 1
                overall_stats['total_responses'] += len(responses)
                
                # Count tokens directly without detailed breakdown
                for response in responses:
                    content = response.get('content', '')
                    if content:
                        token_counts = count_reflective_tokens(content, tokens)
                        for token, count in token_counts.items():
                            overall_stats['total_token_counts'][token] += count
            
            processed_problems += 1
            if processed_problems % 50 == 0:  # Progress update every 50 problems
                print(f"  Processed {processed_problems}/{total_problems} problems...")
        
        # Calculate overall averages
        if overall_stats['total_responses'] > 0:
            if exhaustive:
                overall_stats['avg_total_reflective_tokens_per_response'] = sum(
                    stats['avg_total_reflective_tokens_per_response'] * stats['total_responses']
                    for stats in problem_stats.values()
                ) / overall_stats['total_responses']
            else:
                # Calculate average for simplified mode
                total_reflective_tokens = sum(overall_stats['total_token_counts'].values())
                overall_stats['avg_total_reflective_tokens_per_response'] = total_reflective_tokens / overall_stats['total_responses']
            
            for token in tokens:
                total_count = overall_stats['total_token_counts'][token]
                overall_stats['avg_tokens_per_response'][token] = total_count / overall_stats['total_responses']
        
        # Convert defaultdicts to regular dicts for JSON serialization
        overall_stats['avg_tokens_per_response'] = dict(overall_stats['avg_tokens_per_response'])
        overall_stats['total_token_counts'] = dict(overall_stats['total_token_counts'])
        
        result = {
            'file_path': str(file_path),
            'overall_stats': overall_stats,
            'reflective_tokens_analyzed': tokens
        }
        
        if exhaustive:
            result['problem_stats'] = problem_stats
        
        return result
        
    except Exception as e:
        return {
            'file_path': str(file_path),
            'error': str(e),
            'reflective_tokens_analyzed': tokens
        }


def find_result_files(root_dir: Path) -> List[Path]:
    """
    Recursively find all results.json files in the directory tree.
    
    Args:
        root_dir: Root directory to search
        
    Returns:
        List of paths to results.json files
    """
    result_files = []
    
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file == 'results.json':
                result_files.append(Path(root) / file)
    
    return result_files


def save_results(results: Dict[str, Any], output_path: Path):
    """
    Save analysis results to a JSON file.
    
    Args:
        results: Analysis results to save
        output_path: Path to save the results
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description='Analyze reflective tokens in evaluation results')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory containing evaluation results')
    parser.add_argument('--tokens', type=str, nargs='*', default=REFLECTIVE_TOKENS,
                       help='List of reflective tokens to analyze (default: predefined list)')
    parser.add_argument('--output_suffix', type=str, default='_reflective_token_stats.json',
                       help='Suffix for output files (default: _reflective_token_stats.json)')
    parser.add_argument('--exhaustive', action='store_true',
                       help='Include detailed problem-by-problem statistics (default: simplified output)')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        return 1
    
    # Find all results.json files
    result_files = find_result_files(input_dir)
    print(f"Found {len(result_files)} results.json files to analyze")
    
    if not result_files:
        print("No results.json files found in the specified directory")
        return 1
    
    # Process each file
    all_results = {}
    successful_files = 0
    failed_files = 0
    
    for i, result_file in enumerate(result_files, 1):
        print(f"[{i}/{len(result_files)}] Processing: {result_file}")
        
        # Analyze the file
        analysis_result = process_result_file(result_file, args.tokens, args.exhaustive)
        
        if 'error' in analysis_result:
            print(f"  Error: {analysis_result['error']}")
            failed_files += 1
        else:
            successful_files += 1
            
            # Save individual results
            output_file = result_file.parent / f"{result_file.stem}{args.output_suffix}"
            save_results(analysis_result, output_file)
            print(f"  Saved results to: {output_file}")
            
            # Print summary for this file
            if not args.exhaustive:
                stats = analysis_result['overall_stats']
                print(f"  Summary: {stats['total_problems']} problems, {stats['total_responses']} responses")
                print(f"  Avg reflective tokens per response: {stats['avg_total_reflective_tokens_per_response']:.4f}")
        
        all_results[str(result_file)] = analysis_result
    
    # Save summary
    summary = {
        'input_directory': str(input_dir),
        'reflective_tokens_analyzed': args.tokens,
        'total_files_processed': len(result_files),
        'successful_files': successful_files,
        'failed_files': failed_files,
        'results': all_results
    }
    
    summary_file = input_dir / f"summary{args.output_suffix}"
    save_results(summary, summary_file)
    
    print(f"\nSummary:")
    print(f"  Total files processed: {len(result_files)}")
    print(f"  Successful: {successful_files}")
    print(f"  Failed: {failed_files}")
    print(f"  Summary saved to: {summary_file}")
    
    return 0


if __name__ == '__main__':
    exit(main())
