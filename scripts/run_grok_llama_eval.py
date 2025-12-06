#!/usr/bin/env python3
"""
Run SPICE evaluation on Grok and LLaMA models for 650 problems.
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

# Check API keys
XAI_API_KEY = os.getenv("XAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not XAI_API_KEY:
    print("ERROR: XAI_API_KEY not set")
    sys.exit(1)
if not GROQ_API_KEY:
    print("ERROR: GROQ_API_KEY not set")
    sys.exit(1)

print(f"XAI_API_KEY: {XAI_API_KEY[:20]}...")
print(f"GROQ_API_KEY: {GROQ_API_KEY[:20]}...")

# Use the same evaluator as run_spice_eval_650.py
from scripts.evaluate_with_spice import EnhancedLLMEvaluator

# Models to evaluate
MODELS = [
    ("Grok 3 Mini", "grok-3-mini"),
    ("LLaMA 3.3 70B", "llama-3.3-70b-versatile"),
]

def load_all_problems():
    """Load all 650 problems (500 training + 150 test)"""
    problems = []
    base_path = Path(__file__).parent.parent / "benchmarks"
    
    # Training problems (levels 1-4)
    for level in [1, 2, 3, 4]:
        level_dir = base_path / f"level_{level}"
        for json_file in sorted(level_dir.glob("problems_*.json")):
            with open(json_file) as f:
                data = json.load(f)
                for p in data.get("problems", []):
                    p["level"] = level
                    p["source"] = "training"
                    problems.append(p)
    
    # Test problems
    test_file = base_path / "test_set_v2" / "test_problems_v2.json"
    if test_file.exists():
        with open(test_file) as f:
            data = json.load(f)
            for p in data.get("problems", []):
                p["source"] = "test"
                problems.append(p)
    
    return problems

def call_model(model_name: str, provider: str, prompt: str) -> str:
    """Call the appropriate model API"""
    if provider == "xai":
        response = xai_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=2000
        )
        return response.choices[0].message.content
    elif provider == "groq":
        response = groq_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=2000
        )
        return response.choices[0].message.content
    else:
        raise ValueError(f"Unknown provider: {provider}")

def create_prompt(problem: dict) -> str:
    """Create evaluation prompt from problem"""
    prompt = problem.get("prompt", "")
    if not prompt:
        # Construct from specs
        specs = problem.get("specs", {})
        topology = specs.get("topology", "buck")
        vin = specs.get("vin", 24)
        vout = specs.get("vout", 12)
        iout = specs.get("iout", 2)
        fsw = specs.get("fsw", 100000)
        ripple = specs.get("vout_ripple_pct", 5)
        
        prompt = f"""Design a {topology} DC-DC converter with the following specifications:
- Input voltage: {vin}V
- Output voltage: {vout}V  
- Output current: {iout}A
- Switching frequency: {fsw/1000:.0f}kHz
- Output voltage ripple: <{ripple}%

Provide the component values (L, C, duty cycle) and explain your calculations."""
    
    return prompt

def run_evaluation(model_name: str, provider: str, problems: list, output_file: str):
    """Run SPICE evaluation on all problems"""
    evaluator = PowerConverterEvaluator()
    results = []
    
    # Check for existing progress
    progress_file = output_file.replace(".json", "_progress.json")
    if os.path.exists(progress_file):
        with open(progress_file) as f:
            results = json.load(f)
        print(f"Resuming from {len(results)} completed problems")
    
    completed_ids = {r.get("problem_id") for r in results}
    remaining = [p for p in problems if p.get("id", p.get("problem_id")) not in completed_ids]
    
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name} ({provider})")
    print(f"Total: {len(problems)}, Remaining: {len(remaining)}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    for i, problem in enumerate(remaining):
        problem_id = problem.get("id", problem.get("problem_id", f"P{i}"))
        level = problem.get("level", 0)
        source = problem.get("source", "unknown")
        
        try:
            # Get LLM response
            prompt = create_prompt(problem)
            t0 = time.time()
            response = call_model(model_name, provider, prompt)
            latency = time.time() - t0
            
            # Evaluate with SPICE
            specs = problem.get("specs", {})
            eval_result = evaluator.evaluate_llm_response(response, specs)
            
            result = {
                "problem_id": problem_id,
                "level": level,
                "source": source,
                "success": eval_result.get("success", False),
                "spice_success": eval_result.get("spice_success", False),
                "total_score": eval_result.get("total_score", 0),
                "vout_score": eval_result.get("vout_score", 0),
                "ripple_score": eval_result.get("ripple_score", 0),
                "efficiency_score": eval_result.get("efficiency_score", 0),
                "simulated_vout": eval_result.get("simulated_vout"),
                "target_vout": eval_result.get("target_vout"),
                "vout_error_pct": eval_result.get("vout_error_pct"),
                "ripple_pct": eval_result.get("ripple_pct"),
                "efficiency_pct": eval_result.get("efficiency_pct"),
                "latency": latency,
                "error": eval_result.get("error"),
            }
            
            results.append(result)
            
            # Status
            vout_err = result.get("vout_error_pct", 100)
            status = "✓" if abs(vout_err) < 20 else "✗"
            elapsed = time.time() - start_time
            rate = (i + 1) / (elapsed / 60) if elapsed > 0 else 0
            remaining_time = (len(remaining) - i - 1) / rate if rate > 0 else 0
            
            print(f"[{len(results)}/{len(problems)}] {problem_id} (L{level}, {source})")
            print(f"  {status} Score: {result['total_score']:.1f}/100, Vout: {result.get('simulated_vout', 'N/A')}V (target: {result.get('target_vout', 'N/A')}V)")
            
            if (i + 1) % 10 == 0:
                print(f"  [Progress: {len(results)}/{len(problems)}, {rate:.1f}/min, ~{remaining_time:.0f}min remaining]")
            
            # Save progress
            if (i + 1) % 5 == 0:
                with open(progress_file, "w") as f:
                    json.dump(results, f, indent=2)
                    
        except Exception as e:
            print(f"[{len(results)+1}/{len(problems)}] {problem_id}: ❌ Error: {e}")
            results.append({
                "problem_id": problem_id,
                "level": level,
                "source": source,
                "success": False,
                "error": str(e)
            })
        
        # Rate limiting
        time.sleep(0.5)
    
    # Save final results
    with open(output_file, "w") as f:
        json.dump({"model": model_name, "provider": provider, "results": results}, f, indent=2)
    
    # Remove progress file
    if os.path.exists(progress_file):
        os.remove(progress_file)
    
    # Summary
    correct = sum(1 for r in results if abs(r.get("vout_error_pct", 100)) < 20)
    print(f"\n{'='*60}")
    print(f"COMPLETED: {model_name}")
    print(f"Accuracy: {correct}/{len(results)} ({correct/len(results)*100:.1f}%)")
    print(f"Results saved to: {output_file}")
    print(f"{'='*60}\n")
    
    return results

def main():
    problems = load_all_problems()
    print(f"Loaded {len(problems)} problems")
    
    results_dir = Path(__file__).parent.parent / "benchmarks" / "results"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for model_name, provider in MODELS:
        safe_name = model_name.replace(":", "_").replace("/", "_")
        output_file = results_dir / f"spice_full_{safe_name}_{timestamp}.json"
        
        try:
            run_evaluation(model_name, provider, problems, str(output_file))
        except Exception as e:
            print(f"ERROR evaluating {model_name}: {e}")
            continue

if __name__ == "__main__":
    main()
