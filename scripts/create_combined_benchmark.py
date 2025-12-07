#!/usr/bin/env python3
"""
Create Combined Benchmark: 400 Train + 100 Test
Mix of Expert-Verified (conceptual) and Synthetic (design) problems across all levels.

Distribution:
- Expert-verified: ~40% (conceptual/MCQ from GATE+MIT)  
- Synthetic: ~60% (circuit design with SPICE specs)

Train: 400 problems
Test: 100 problems (held out)
"""

import json
import random
from pathlib import Path
from typing import List, Dict

random.seed(42)

PROJECT_ROOT = Path(__file__).parent.parent
EXPERT_DIR = PROJECT_ROOT / "benchmarks" / "expert_verified" / "train_test_split"
SYNTHETIC_DIR = PROJECT_ROOT / "benchmarks"
OUTPUT_DIR = PROJECT_ROOT / "benchmarks" / "combined_benchmark"


def load_expert_verified() -> tuple:
    """Load expert-verified problems (GATE + MIT)"""
    with open(EXPERT_DIR / "combined_train.json") as f:
        train_data = json.load(f)
    with open(EXPERT_DIR / "combined_test.json") as f:
        test_data = json.load(f)
    
    # Add problem_type marker
    for p in train_data["problems"]:
        p["benchmark_type"] = "expert_verified"
        p["evaluation_method"] = "conceptual"  # MCQ/calculation
    for p in test_data["problems"]:
        p["benchmark_type"] = "expert_verified"
        p["evaluation_method"] = "conceptual"
    
    return train_data["problems"], test_data["problems"]


def load_synthetic_problems() -> Dict[int, List[Dict]]:
    """Load synthetic design problems by level"""
    problems_by_level = {}
    
    for level in range(1, 6):
        level_dir = SYNTHETIC_DIR / f"level_{level}"
        if not level_dir.exists():
            continue
        
        problems = []
        for prob_file in sorted(level_dir.glob("problems_*.json")):
            with open(prob_file) as f:
                data = json.load(f)
                for p in data.get("problems", []):
                    p["level"] = level
                    p["benchmark_type"] = "synthetic"
                    p["evaluation_method"] = "spice"  # Can be SPICE simulated
                    problems.append(p)
        
        problems_by_level[level] = problems
    
    return problems_by_level


def create_combined_benchmark():
    """Create 400 train + 100 test combined benchmark"""
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load all sources
    expert_train, expert_test = load_expert_verified()
    synthetic_by_level = load_synthetic_problems()
    
    print("="*60)
    print("SOURCE DATA")
    print("="*60)
    print(f"Expert Train: {len(expert_train)}")
    print(f"Expert Test: {len(expert_test)}")
    for lvl, probs in synthetic_by_level.items():
        print(f"Synthetic L{lvl}: {len(probs)}")
    
    # Target distribution for 500 total (400 train + 100 test)
    # Expert: ~40% = 200 (160 train + 40 test)
    # Synthetic: ~60% = 300 (240 train + 60 test)
    
    # Combine all expert problems and shuffle
    all_expert = expert_train + expert_test
    random.shuffle(all_expert)
    
    # Take 160 for train, 40 for test from expert
    expert_for_train = all_expert[:160]
    expert_for_test = all_expert[160:200]
    
    # Combine all synthetic problems
    all_synthetic = []
    for level, probs in synthetic_by_level.items():
        all_synthetic.extend(probs)
    random.shuffle(all_synthetic)
    
    # Take 240 for train, 60 for test from synthetic
    synthetic_for_train = all_synthetic[:240]
    synthetic_for_test = all_synthetic[240:300]
    
    # Combine and shuffle
    train_problems = expert_for_train + synthetic_for_train
    test_problems = expert_for_test + synthetic_for_test
    random.shuffle(train_problems)
    random.shuffle(test_problems)
    
    # Count by level and type
    def count_stats(problems):
        stats = {
            "by_level": {},
            "by_type": {"expert_verified": 0, "synthetic": 0},
            "by_eval_method": {"conceptual": 0, "spice": 0}
        }
        for p in problems:
            lvl = p.get("level", 0)
            btype = p.get("benchmark_type", "unknown")
            emethod = p.get("evaluation_method", "unknown")
            
            stats["by_level"][lvl] = stats["by_level"].get(lvl, 0) + 1
            stats["by_type"][btype] = stats["by_type"].get(btype, 0) + 1
            stats["by_eval_method"][emethod] = stats["by_eval_method"].get(emethod, 0) + 1
        return stats
    
    train_stats = count_stats(train_problems)
    test_stats = count_stats(test_problems)
    
    # Save train set
    train_output = {
        "metadata": {
            "name": "PowerElecLLM Combined Benchmark - Training Set",
            "description": "400 problems combining expert-verified (GATE+MIT) and synthetic design problems",
            "total_problems": len(train_problems),
            "split": "train",
            "composition": {
                "expert_verified": train_stats["by_type"]["expert_verified"],
                "synthetic": train_stats["by_type"]["synthetic"]
            },
            "evaluation_methods": train_stats["by_eval_method"],
            "by_level": train_stats["by_level"]
        },
        "problems": train_problems
    }
    
    with open(OUTPUT_DIR / "train_400.json", "w") as f:
        json.dump(train_output, f, indent=2)
    
    # Save test set
    test_output = {
        "metadata": {
            "name": "PowerElecLLM Combined Benchmark - Test Set",
            "description": "100 problems (held out) combining expert-verified and synthetic design problems",
            "total_problems": len(test_problems),
            "split": "test",
            "composition": {
                "expert_verified": test_stats["by_type"]["expert_verified"],
                "synthetic": test_stats["by_type"]["synthetic"]
            },
            "evaluation_methods": test_stats["by_eval_method"],
            "by_level": test_stats["by_level"]
        },
        "problems": test_problems
    }
    
    with open(OUTPUT_DIR / "test_100.json", "w") as f:
        json.dump(test_output, f, indent=2)
    
    # Save summary
    summary = {
        "benchmark_name": "PowerElecLLM Combined Benchmark",
        "random_seed": 42,
        "total_problems": 500,
        "train": {
            "file": "train_400.json",
            "count": 400,
            "expert_verified": train_stats["by_type"]["expert_verified"],
            "synthetic": train_stats["by_type"]["synthetic"],
            "conceptual_eval": train_stats["by_eval_method"]["conceptual"],
            "spice_eval": train_stats["by_eval_method"]["spice"],
            "by_level": train_stats["by_level"]
        },
        "test": {
            "file": "test_100.json",
            "count": 100,
            "expert_verified": test_stats["by_type"]["expert_verified"],
            "synthetic": test_stats["by_type"]["synthetic"],
            "conceptual_eval": test_stats["by_eval_method"]["conceptual"],
            "spice_eval": test_stats["by_eval_method"]["spice"],
            "by_level": test_stats["by_level"]
        },
        "sources": {
            "expert_verified": {
                "GATE_EE": "Official GATE papers 2007-2023 (IISc archive)",
                "MIT_6334": "MIT OCW Power Electronics Spring 2007",
                "license": "Public Domain (GATE) / CC BY-NC-SA 4.0 (MIT)"
            },
            "synthetic": {
                "description": "Algorithmically generated circuit design problems",
                "evaluation": "SPICE simulation with waveform comparison"
            }
        },
        "usage": {
            "train": "Use for fine-tuning, few-shot examples, or in-context learning",
            "test": "Held out for final evaluation - DO NOT use for training"
        }
    }
    
    with open(OUTPUT_DIR / "benchmark_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("COMBINED BENCHMARK CREATED")
    print("="*60)
    print(f"\nTRAIN SET (400 problems):")
    print(f"  Expert-verified (conceptual): {train_stats['by_type']['expert_verified']}")
    print(f"  Synthetic (SPICE): {train_stats['by_type']['synthetic']}")
    print(f"  By level: {train_stats['by_level']}")
    
    print(f"\nTEST SET (100 problems):")
    print(f"  Expert-verified (conceptual): {test_stats['by_type']['expert_verified']}")
    print(f"  Synthetic (SPICE): {test_stats['by_type']['synthetic']}")
    print(f"  By level: {test_stats['by_level']}")
    
    print(f"\nFiles saved to: {OUTPUT_DIR}")
    print("  - train_400.json")
    print("  - test_100.json")
    print("  - benchmark_summary.json")


if __name__ == "__main__":
    create_combined_benchmark()
