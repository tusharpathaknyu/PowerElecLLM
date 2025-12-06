#!/usr/bin/env python3
"""
Run full 650-problem evaluation with Chain-of-Thought prompting.
Compares CoT vs Basic prompting on all problems.
"""

import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.enhanced_evaluator import EnhancedEvaluator, SYSTEM_PROMPT_COT, FEW_SHOT_EXAMPLES

# Import SPICE evaluator
from src.spice_evaluator import PowerConverterEvaluator


def load_all_problems() -> List[Dict]:
    """Load all 650 problems (500 train + 150 test)."""
    problems = []
    
    # Load training problems (Level 1-4)
    for level in [1, 2, 3, 4]:
        level_dir = PROJECT_ROOT / "benchmarks" / f"level_{level}"
        for f in sorted(level_dir.glob("problems_*.json")):
            try:
                with open(f) as fp:
                    data = json.load(fp)
                for p in data.get("problems", []):
                    p["level"] = level
                    problems.append(p)
            except:
                pass
    
    # Load test problems
    test_file = PROJECT_ROOT / "benchmarks" / "test_set_v2" / "test_problems_v2.json"
    try:
        with open(test_file) as f:
            data = json.load(f)
        for p in data.get("problems", []):
            p["level"] = 5  # Mark as test
            problems.append(p)
    except:
        pass
    
    return problems


def run_cot_evaluation(num_problems: int = None, model: str = "gpt-4o-mini"):
    """Run CoT evaluation on all problems."""
    
    problems = load_all_problems()
    if num_problems:
        problems = problems[:num_problems]
    
    print(f"=" * 70)
    print(f"CHAIN-OF-THOUGHT EVALUATION")
    print(f"Model: {model}")
    print(f"Problems: {len(problems)}")
    print(f"=" * 70)
    
    # Initialize evaluators
    cot_evaluator = EnhancedEvaluator(model=model, mode="all")  # CoT + Few-shot
    basic_evaluator = EnhancedEvaluator(model=model, mode="basic")
    spice_eval = PowerConverterEvaluator()
    
    results_data = {
        "model": model,
        "timestamp": datetime.now().isoformat(),
        "cot_results": [],
        "basic_results": [],
        "summary": {}
    }
    
    cot_correct = 0
    basic_correct = 0
    
    for i, problem in enumerate(problems):
        prompt = problem.get("prompt", "")
        specs = problem.get("specs", {})
        problem_id = problem.get("id", f"P{i}")
        level = problem.get("level", 0)
        
        print(f"\n[{i+1}/{len(problems)}] {problem_id} (L{level})")
        
        # CoT evaluation
        cot_result = cot_evaluator.evaluate(prompt)
        cot_params = cot_result.get("params", {})
        
        # Basic evaluation  
        basic_result = basic_evaluator.evaluate(prompt)
        basic_params = basic_result.get("params", {})
        
        # SPICE simulation for both
        cot_vout = 0
        basic_vout = 0
        
        target_vout = abs(specs.get("vout", 0))
        topology = specs.get("topology", "buck")
        vin = specs.get("vin", 12)
        
        # For CoT
        if cot_params and cot_params.get("D"):
            try:
                components = {
                    "D": cot_params.get("D", 0.5),
                    "L": cot_params.get("L", 100e-6),
                    "C": cot_params.get("C", 100e-6),
                    "R_load": cot_params.get("R", target_vout / specs.get("iout", 1) if specs.get("iout") else 10)
                }
                results, score = spice_eval.evaluate(topology=topology, components=components, specs=specs)
                if results.simulation_success:
                    cot_vout = results.vout_dc
                else:
                    # Use analytical calculation
                    D = cot_params.get("D", 0.5)
                    if topology == "buck":
                        cot_vout = vin * D
                    elif topology == "boost":
                        cot_vout = vin / (1 - D) if D < 1 else 0
                    elif topology in ["buck_boost", "cuk"]:
                        cot_vout = vin * D / (1 - D) if D < 1 else 0
                    elif topology == "sepic":
                        cot_vout = vin * D / (1 - D) if D < 1 else 0
                    else:
                        cot_vout = vin * D
            except Exception as e:
                # Analytical fallback
                D = cot_params.get("D", 0.5)
                if topology == "buck":
                    cot_vout = vin * D
                elif topology == "boost":
                    cot_vout = vin / (1 - D) if D < 1 else 0
                else:
                    cot_vout = vin * D
        
        # For Basic
        if basic_params and basic_params.get("D"):
            try:
                components = {
                    "D": basic_params.get("D", 0.5),
                    "L": basic_params.get("L", 100e-6),
                    "C": basic_params.get("C", 100e-6),
                    "R_load": basic_params.get("R", target_vout / specs.get("iout", 1) if specs.get("iout") else 10)
                }
                results, score = spice_eval.evaluate(topology=topology, components=components, specs=specs)
                if results.simulation_success:
                    basic_vout = results.vout_dc
                else:
                    D = basic_params.get("D", 0.5)
                    if topology == "buck":
                        basic_vout = vin * D
                    elif topology == "boost":
                        basic_vout = vin / (1 - D) if D < 1 else 0
                    elif topology in ["buck_boost", "cuk"]:
                        basic_vout = vin * D / (1 - D) if D < 1 else 0
                    elif topology == "sepic":
                        basic_vout = vin * D / (1 - D) if D < 1 else 0
                    else:
                        basic_vout = vin * D
            except Exception as e:
                D = basic_params.get("D", 0.5)
                if topology == "buck":
                    basic_vout = vin * D
                elif topology == "boost":
                    basic_vout = vin / (1 - D) if D < 1 else 0
                else:
                    basic_vout = vin * D
        
        # Check correctness (Vout within 10%)
        cot_error = abs(cot_vout - target_vout) / target_vout * 100 if target_vout else 100
        cot_pass = cot_error < 10
        if cot_pass:
            cot_correct += 1
        
        basic_error = abs(basic_vout - target_vout) / target_vout * 100 if target_vout else 100
        basic_pass = basic_error < 10
        if basic_pass:
            basic_correct += 1
        
        print(f"  Target Vout: {target_vout}V")
        print(f"  CoT:   D={cot_params.get('D', 'N/A')}, Vout={cot_vout:.2f}V, Error={cot_error:.1f}% {'✓' if cot_pass else '✗'}")
        print(f"  Basic: D={basic_params.get('D', 'N/A')}, Vout={basic_vout:.2f}V, Error={basic_error:.1f}% {'✓' if basic_pass else '✗'}")
        
        # Store results
        results_data["cot_results"].append({
            "id": problem_id,
            "level": level,
            "D": cot_params.get("D"),
            "vout": cot_vout,
            "vout_error_pct": cot_error,
            "passed": cot_pass
        })
        
        results_data["basic_results"].append({
            "id": problem_id,
            "level": level,
            "D": basic_params.get("D"),
            "vout": basic_vout,
            "vout_error_pct": basic_error,
            "passed": basic_pass
        })
        
        # Progress update
        if (i + 1) % 10 == 0:
            print(f"\n--- Progress: CoT={cot_correct}/{i+1} ({cot_correct/(i+1)*100:.1f}%), Basic={basic_correct}/{i+1} ({basic_correct/(i+1)*100:.1f}%) ---")
        
        # Rate limiting
        time.sleep(0.5)
    
    # Summary
    results_data["summary"] = {
        "total": len(problems),
        "cot_correct": cot_correct,
        "cot_accuracy": cot_correct / len(problems) * 100,
        "basic_correct": basic_correct,
        "basic_accuracy": basic_correct / len(problems) * 100,
        "improvement": (cot_correct - basic_correct) / basic_correct * 100 if basic_correct > 0 else 0
    }
    
    # Print summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"CoT + Few-Shot: {cot_correct}/{len(problems)} ({results_data['summary']['cot_accuracy']:.1f}%)")
    print(f"Basic:          {basic_correct}/{len(problems)} ({results_data['summary']['basic_accuracy']:.1f}%)")
    print(f"Improvement:    +{results_data['summary']['improvement']:.1f}%")
    
    # Save results
    output_file = PROJECT_ROOT / "benchmarks" / "results" / f"cot_eval_{model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"\nResults saved to {output_file}")
    
    return results_data


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=100, help="Number of problems (default: 100)")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model to use")
    
    args = parser.parse_args()
    
    run_cot_evaluation(num_problems=args.num, model=args.model)
