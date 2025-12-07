#!/usr/bin/env python3
"""
Create 80/20 train/test split for expert-verified benchmark problems.
Splits each difficulty level separately to maintain distribution.
"""

import json
import os
import random
from pathlib import Path

# Set seed for reproducibility
random.seed(42)

# Paths
BASE_DIR = Path(__file__).parent.parent
EXPERT_DIR = BASE_DIR / "benchmarks" / "expert_verified"
BY_LEVEL_DIR = EXPERT_DIR / "by_level"
TRAIN_TEST_DIR = EXPERT_DIR / "train_test_split"

def load_level_problems(level: int) -> dict:
    """Load problems for a specific level."""
    filepath = BY_LEVEL_DIR / f"level_{level}_problems.json"
    with open(filepath, 'r') as f:
        return json.load(f)

def split_problems(problems: list, train_ratio: float = 0.8) -> tuple:
    """Split problems into train and test sets."""
    # Shuffle for random split
    shuffled = problems.copy()
    random.shuffle(shuffled)
    
    split_idx = int(len(shuffled) * train_ratio)
    train_set = shuffled[:split_idx]
    test_set = shuffled[split_idx:]
    
    return train_set, test_set

def create_split():
    """Create train/test split for all levels."""
    # Create output directory
    TRAIN_TEST_DIR.mkdir(parents=True, exist_ok=True)
    
    all_train = []
    all_test = []
    level_stats = {}
    
    for level in range(1, 5):  # L1 to L4
        print(f"\nProcessing Level {level}...")
        
        # Load problems
        data = load_level_problems(level)
        problems = data['problems']
        total = len(problems)
        
        # Split
        train, test = split_problems(problems)
        
        # Save level-specific files
        train_data = {
            "source": f"Expert-Verified (GATE + MIT) - Training Set",
            "level": level,
            "difficulty": data['difficulty'],
            "total_problems": len(train),
            "split": "train",
            "split_ratio": "80/20",
            "problems": train
        }
        
        test_data = {
            "source": f"Expert-Verified (GATE + MIT) - Test Set",
            "level": level,
            "difficulty": data['difficulty'],
            "total_problems": len(test),
            "split": "test",
            "split_ratio": "80/20",
            "problems": test
        }
        
        # Save individual level files
        train_file = TRAIN_TEST_DIR / f"level_{level}_train.json"
        test_file = TRAIN_TEST_DIR / f"level_{level}_test.json"
        
        with open(train_file, 'w') as f:
            json.dump(train_data, f, indent=2)
        
        with open(test_file, 'w') as f:
            json.dump(test_data, f, indent=2)
        
        # Accumulate
        all_train.extend(train)
        all_test.extend(test)
        
        level_stats[f"L{level}"] = {
            "total": total,
            "train": len(train),
            "test": len(test)
        }
        
        print(f"  Level {level}: {total} total → {len(train)} train / {len(test)} test")
    
    # Save combined files
    combined_train = {
        "source": "Expert-Verified (GATE + MIT) - Combined Training Set",
        "description": "80% of problems from each difficulty level",
        "total_problems": len(all_train),
        "split": "train",
        "by_level": {f"L{i}": level_stats[f"L{i}"]["train"] for i in range(1, 5)},
        "problems": all_train
    }
    
    combined_test = {
        "source": "Expert-Verified (GATE + MIT) - Combined Test Set",
        "description": "20% of problems from each difficulty level (held out for evaluation)",
        "total_problems": len(all_test),
        "split": "test",
        "by_level": {f"L{i}": level_stats[f"L{i}"]["test"] for i in range(1, 5)},
        "problems": all_test
    }
    
    with open(TRAIN_TEST_DIR / "combined_train.json", 'w') as f:
        json.dump(combined_train, f, indent=2)
    
    with open(TRAIN_TEST_DIR / "combined_test.json", 'w') as f:
        json.dump(combined_test, f, indent=2)
    
    # Save summary
    summary = {
        "split_ratio": "80/20",
        "random_seed": 42,
        "total_problems": {
            "train": len(all_train),
            "test": len(all_test),
            "total": len(all_train) + len(all_test)
        },
        "by_level": level_stats,
        "files": {
            "train": [
                "level_1_train.json",
                "level_2_train.json",
                "level_3_train.json",
                "level_4_train.json",
                "combined_train.json"
            ],
            "test": [
                "level_1_test.json",
                "level_2_test.json",
                "level_3_test.json",
                "level_4_test.json",
                "combined_test.json"
            ]
        },
        "usage": {
            "train": "Use for fine-tuning or few-shot examples",
            "test": "Hold out for evaluation - do not use for training"
        }
    }
    
    with open(TRAIN_TEST_DIR / "split_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*50}")
    print("TRAIN/TEST SPLIT SUMMARY")
    print(f"{'='*50}")
    print(f"Total: {len(all_train) + len(all_test)} problems")
    print(f"Train: {len(all_train)} (80%)")
    print(f"Test:  {len(all_test)} (20%)")
    print(f"\nBy Level:")
    for level in range(1, 5):
        stats = level_stats[f"L{level}"]
        print(f"  L{level}: {stats['train']} train / {stats['test']} test")
    print(f"\nFiles saved to: {TRAIN_TEST_DIR}")
    
    return summary

if __name__ == "__main__":
    create_split()
