#!/usr/bin/env python3
"""
Run evaluation of all models on the 150-problem test set.

Models to evaluate:
1. Fine-tuned GPT-4o (best performer on training set)
2. Fine-tuned GPT-4o-mini
3. GPT-4o base
4. Grok 4.1 Fast
5. LLaMA 3.3 70B (via Groq - free)

Estimated costs:
- GPT-4o FT: ~$0.50
- GPT-4o-mini FT: ~$0.05
- GPT-4o base: ~$0.50
- Grok 4.1: ~$0.30
- LLaMA: FREE

Total: ~$1.35
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime

# Models to evaluate
MODELS = [
    {
        "name": "Fine-tuned GPT-4o",
        "model_id": "ft:gpt-4o-2024-08-06:personal:powerelec:CjLJYOod",
        "api_type": "openai",
        "estimated_cost": 0.50
    },
    {
        "name": "Fine-tuned GPT-4o-mini",
        "model_id": "ft:gpt-4o-mini-2024-07-18:personal:powerelec:CjJntxcS",
        "api_type": "openai",
        "estimated_cost": 0.05
    },
    {
        "name": "GPT-4o (base)",
        "model_id": "gpt-4o",
        "api_type": "openai",
        "estimated_cost": 0.50
    },
    {
        "name": "Grok 4.1 Fast",
        "model_id": "grok-3-fast",
        "api_type": "xai",
        "env_var": "XAI_API_KEY",
        "estimated_cost": 0.30
    },
    {
        "name": "LLaMA 3.3 70B",
        "model_id": "llama-3.3-70b-versatile",
        "api_type": "groq",
        "env_var": "GROQ_API_KEY",
        "estimated_cost": 0.00
    }
]

def run_evaluation(model_config, test_file, output_dir):
    """Run evaluation for a single model"""
    
    model_id = model_config["model_id"]
    name = model_config["name"]
    
    print(f"\n{'='*60}")
    print(f"Evaluating: {name}")
    print(f"Model ID: {model_id}")
    print(f"Estimated cost: ${model_config['estimated_cost']:.2f}")
    print(f"{'='*60}")
    
    # Build output filename
    safe_name = model_id.replace("/", "_").replace(":", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"testset_eval_{safe_name}_{timestamp}.json"
    
    # Build command
    cmd = [
        "python", "scripts/evaluate_with_spice.py",
        "--model", model_id,
        "--test-file", str(test_file),
        "--output", str(output_file)
    ]
    
    # Set environment variables if needed
    env = os.environ.copy()
    if "env_var" in model_config:
        var_name = model_config["env_var"]
        if var_name not in env:
            print(f"Warning: {var_name} not set, skipping {name}")
            return None
    
    # Run evaluation
    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=False,
            text=True,
            cwd=Path(__file__).parent.parent
        )
        
        if result.returncode != 0:
            print(f"Error running evaluation for {name}")
            return None
            
        return output_file
        
    except Exception as e:
        print(f"Error: {e}")
        return None


def summarize_results(output_dir):
    """Summarize results from all evaluations"""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE TEST SET RESULTS")
    print("="*80)
    
    results_files = list(output_dir.glob("testset_eval_*.json"))
    
    all_results = []
    
    for f in results_files:
        try:
            with open(f) as fp:
                data = json.load(fp)
            
            model = data.get("model", "Unknown")
            summary = data.get("summary", {})
            
            all_results.append({
                "model": model,
                "total": summary.get("total", 0),
                "passed": summary.get("passed", 0),
                "accuracy": summary.get("overall_accuracy", 0),
                "by_level": summary.get("by_level", {})
            })
        except Exception as e:
            print(f"Error reading {f}: {e}")
    
    # Sort by accuracy
    all_results.sort(key=lambda x: x["accuracy"], reverse=True)
    
    # Print summary table
    print("\n" + "-"*80)
    print(f"{'Model':<45} {'Total':<10} {'Passed':<10} {'Accuracy':<10}")
    print("-"*80)
    
    for r in all_results:
        model_short = r["model"][:42] + "..." if len(r["model"]) > 45 else r["model"]
        print(f"{model_short:<45} {r['total']:<10} {r['passed']:<10} {r['accuracy']:.1f}%")
    
    # Print per-level breakdown
    print("\n" + "-"*80)
    print("Per-Level Breakdown:")
    print("-"*80)
    
    for r in all_results:
        print(f"\n{r['model'][:50]}:")
        for level, data in sorted(r["by_level"].items()):
            avg_score = data.get("avg_score", 0)
            print(f"  {level}: {data['passed']}/{data['total']} ({data['accuracy']:.1f}%) - Avg Score: {avg_score:.1f}")
    
    # Save summary
    summary_file = output_dir / f"test_set_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "results": all_results
        }, f, indent=2)
    
    print(f"\nSummary saved to: {summary_file}")


def main():
    project_root = Path(__file__).parent.parent
    test_file = project_root / "benchmarks" / "test_set" / "test_problems.json"
    output_dir = project_root / "benchmarks" / "results"
    
    if not test_file.exists():
        print(f"Error: Test file not found: {test_file}")
        sys.exit(1)
    
    # Load test file to show stats
    with open(test_file) as f:
        test_data = json.load(f)
    
    n_problems = len(test_data.get("problems", []))
    print(f"Test set: {n_problems} problems")
    
    # Calculate total estimated cost
    total_cost = sum(m["estimated_cost"] for m in MODELS)
    print(f"Estimated total cost: ${total_cost:.2f}")
    
    # Confirm
    print("\nModels to evaluate:")
    for m in MODELS:
        print(f"  - {m['name']} (${m['estimated_cost']:.2f})")
    
    response = input("\nProceed with evaluation? [y/N]: ")
    if response.lower() != 'y':
        print("Aborted.")
        sys.exit(0)
    
    # Run evaluations
    output_files = []
    for model_config in MODELS:
        output_file = run_evaluation(model_config, test_file, output_dir)
        if output_file:
            output_files.append(output_file)
    
    # Summarize results
    summarize_results(output_dir)


if __name__ == "__main__":
    main()
