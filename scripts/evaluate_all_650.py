#!/usr/bin/env python3
"""
Comprehensive evaluation of all models on FULL 650-problem dataset
(500 training + 150 test problems)

This provides complete benchmark results for the paper.

Models:
1. Fine-tuned GPT-4o
2. Fine-tuned GPT-4o-mini
3. GPT-4o (base)
4. Grok 4.1 Fast
5. LLaMA 3.3 70B (Groq - FREE)

Estimated costs: ~$8-10 total for all 650 problems × 5 models
"""

import os
import sys
import json
import time
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI


class ComprehensiveEvaluator:
    """Evaluate LLMs on power electronics problems"""
    
    def __init__(self, model: str, api_type: str = "openai"):
        self.model = model
        self.api_type = api_type
        
        if api_type == "xai":
            self.client = OpenAI(
                api_key=os.environ.get("XAI_API_KEY"),
                base_url="https://api.x.ai/v1"
            )
        elif api_type == "groq":
            self.client = OpenAI(
                api_key=os.environ.get("GROQ_API_KEY"),
                base_url="https://api.groq.com/openai/v1"
            )
        else:
            self.client = OpenAI()
        
        self.system_prompt = """You are an expert power electronics engineer. 
When given a converter design problem:
1. Identify the topology (buck, boost, buck-boost, flyback, etc.)
2. Calculate the duty cycle using the correct equation
3. Calculate inductor and capacitor values for the given specifications
4. Verify your answer by substituting back

Always provide numerical values for:
- Output voltage (Vout) in Volts
- Duty cycle (D) as a decimal between 0 and 1
- Inductor value (L) with units (µH or mH)
- Capacitor value (C) with units (µF or mF)

Show your calculations step by step."""

    def call_llm(self, prompt: str, retries: int = 3) -> str:
        """Call LLM with retries"""
        for attempt in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=2000
                )
                return response.choices[0].message.content
            except Exception as e:
                if attempt < retries - 1:
                    wait_time = (attempt + 1) * 5
                    print(f"  API error, retrying in {wait_time}s: {str(e)[:50]}")
                    time.sleep(wait_time)
                else:
                    print(f"  API failed after {retries} attempts: {str(e)[:50]}")
                    return ""
        return ""

    def parse_vout(self, response: str) -> Optional[float]:
        """Extract Vout from LLM response"""
        patterns = [
            r"[Vv]out\s*[=:≈]\s*([\d.]+)\s*V",
            r"output voltage[=:\s]*([\d.]+)\s*V",
            r"[Vv]_?out\s*=\s*([\d.]+)",
            r"([\d.]+)\s*V\s*(?:output|DC|out)",
        ]
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except:
                    continue
        return None

    def evaluate_problem(self, problem: Dict) -> Dict:
        """Evaluate a single problem"""
        prompt = problem.get("prompt", "")
        level = problem.get("level", 1)
        target_vout = problem.get("specs", {}).get("vout")
        
        start_time = time.time()
        response = self.call_llm(prompt)
        latency = time.time() - start_time
        
        if not response:
            return {
                "problem_id": problem.get("id"),
                "level": level,
                "success": False,
                "error": "No response",
                "latency": latency
            }
        
        predicted_vout = self.parse_vout(response)
        
        if predicted_vout is None or target_vout is None:
            return {
                "problem_id": problem.get("id"),
                "level": level,
                "success": False,
                "error": "Could not parse Vout",
                "predicted_vout": predicted_vout,
                "target_vout": target_vout,
                "latency": latency
            }
        
        error_pct = abs(predicted_vout - target_vout) / target_vout * 100
        success = error_pct < 5  # 5% tolerance
        
        return {
            "problem_id": problem.get("id"),
            "level": level,
            "success": success,
            "predicted_vout": predicted_vout,
            "target_vout": target_vout,
            "error_pct": error_pct,
            "latency": latency
        }


def load_all_problems(project_root: Path) -> List[Dict]:
    """Load all 650 problems (500 training + 150 test)"""
    all_problems = []
    
    # Load training problems (500)
    for level in [1, 2, 3, 4]:
        level_dir = project_root / "benchmarks" / f"level_{level}"
        if level_dir.exists():
            for f in level_dir.glob("problems_*.json"):
                try:
                    with open(f) as fp:
                        data = json.load(fp)
                    for p in data.get("problems", []):
                        p["level"] = level
                        p["source"] = "training"
                        all_problems.append(p)
                except Exception as e:
                    print(f"Error loading {f}: {e}")
    
    # Load test problems (150)
    test_file = project_root / "benchmarks" / "test_set" / "test_problems.json"
    if test_file.exists():
        try:
            with open(test_file) as f:
                data = json.load(f)
            for p in data.get("problems", []):
                p["source"] = "test"
                all_problems.append(p)
        except Exception as e:
            print(f"Error loading test set: {e}")
    
    return all_problems


def run_model_evaluation(
    model_config: Dict,
    problems: List[Dict],
    output_dir: Path,
    resume_file: Optional[Path] = None
) -> Dict:
    """Run evaluation for a single model with progress saving"""
    
    model_id = model_config["model_id"]
    model_name = model_config["name"]
    api_type = model_config.get("api_type", "openai")
    
    print(f"\n{'='*70}")
    print(f"Model: {model_name}")
    print(f"Problems: {len(problems)}")
    print(f"Estimated cost: ${model_config.get('estimated_cost', 0):.2f}")
    print(f"{'='*70}")
    
    # Check for resume file
    results = []
    completed_ids = set()
    
    if resume_file and resume_file.exists():
        try:
            with open(resume_file) as f:
                existing = json.load(f)
            results = existing.get("results", [])
            completed_ids = {r["problem_id"] for r in results}
            print(f"Resuming from {len(results)} completed problems")
        except:
            pass
    
    # Filter problems to evaluate
    remaining = [p for p in problems if p.get("id") not in completed_ids]
    print(f"Remaining: {len(remaining)}")
    
    if not remaining:
        print("All problems already evaluated!")
        return {"results": results}
    
    # Create evaluator
    evaluator = ComprehensiveEvaluator(model_id, api_type)
    
    # Progress file
    safe_name = model_id.replace("/", "_").replace(":", "_")
    progress_file = output_dir / f"progress_{safe_name}.json"
    
    # Evaluate
    total = len(problems)
    for i, problem in enumerate(remaining):
        current = len(results) + 1
        
        result = evaluator.evaluate_problem(problem)
        result["source"] = problem.get("source", "unknown")
        results.append(result)
        
        # Progress indicator
        status = "✓" if result.get("success") else "✗"
        err = f" ({result.get('error_pct', 0):.1f}%)" if result.get("error_pct") else ""
        print(f"[{current}/{total}] {result.get('problem_id', 'N/A')} {status}{err}")
        
        # Save progress every 10 problems
        if current % 10 == 0:
            with open(progress_file, "w") as f:
                json.dump({"model": model_id, "results": results}, f)
        
        # Rate limiting
        if api_type == "groq":
            time.sleep(1.5)  # Groq rate limit
        else:
            time.sleep(0.2)
    
    return {"model": model_id, "results": results}


def calculate_summary(results: List[Dict]) -> Dict:
    """Calculate summary statistics"""
    
    by_level = {}
    by_source = {"training": {"total": 0, "passed": 0}, "test": {"total": 0, "passed": 0}}
    
    for r in results:
        lvl = r.get("level", 1)
        source = r.get("source", "training")
        
        if lvl not in by_level:
            by_level[lvl] = {"total": 0, "passed": 0}
        
        by_level[lvl]["total"] += 1
        by_source[source]["total"] += 1
        
        if r.get("success"):
            by_level[lvl]["passed"] += 1
            by_source[source]["passed"] += 1
    
    total = len(results)
    passed = sum(1 for r in results if r.get("success"))
    
    return {
        "total": total,
        "passed": passed,
        "accuracy": passed / total * 100 if total > 0 else 0,
        "by_level": {
            f"L{k}": {
                "total": v["total"],
                "passed": v["passed"],
                "accuracy": v["passed"] / v["total"] * 100 if v["total"] > 0 else 0
            }
            for k, v in sorted(by_level.items())
        },
        "by_source": {
            k: {
                "total": v["total"],
                "passed": v["passed"],
                "accuracy": v["passed"] / v["total"] * 100 if v["total"] > 0 else 0
            }
            for k, v in by_source.items()
        }
    }


def main():
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "benchmarks" / "results"
    output_dir.mkdir(exist_ok=True)
    
    # Models to evaluate
    models = [
        {
            "name": "Fine-tuned GPT-4o",
            "model_id": "ft:gpt-4o-2024-08-06:personal:powerelec:CjLJYOod",
            "api_type": "openai",
            "estimated_cost": 1.50
        },
        {
            "name": "Fine-tuned GPT-4o-mini",
            "model_id": "ft:gpt-4o-mini-2024-07-18:personal:powerelec:CjJntxcS",
            "api_type": "openai",
            "estimated_cost": 0.15
        },
        {
            "name": "GPT-4o (base)",
            "model_id": "gpt-4o",
            "api_type": "openai",
            "estimated_cost": 1.50
        },
        {
            "name": "Grok 4.1 Fast",
            "model_id": "grok-3-fast",
            "api_type": "xai",
            "estimated_cost": 0.80
        },
        {
            "name": "LLaMA 3.3 70B",
            "model_id": "llama-3.3-70b-versatile",
            "api_type": "groq",
            "estimated_cost": 0.00
        }
    ]
    
    # Load all problems
    print("Loading problems...")
    all_problems = load_all_problems(project_root)
    print(f"Total problems: {len(all_problems)}")
    
    # Count by source
    training = [p for p in all_problems if p.get("source") == "training"]
    test = [p for p in all_problems if p.get("source") == "test"]
    print(f"  Training: {len(training)}")
    print(f"  Test: {len(test)}")
    
    # Check API keys
    missing_keys = []
    if not os.environ.get("OPENAI_API_KEY"):
        missing_keys.append("OPENAI_API_KEY")
    if not os.environ.get("XAI_API_KEY"):
        missing_keys.append("XAI_API_KEY")
    if not os.environ.get("GROQ_API_KEY"):
        missing_keys.append("GROQ_API_KEY")
    
    if missing_keys:
        print(f"\n⚠️  Missing API keys: {', '.join(missing_keys)}")
        print("Some models may be skipped.")
    
    # Estimated total cost
    total_cost = sum(m["estimated_cost"] for m in models)
    print(f"\nEstimated total cost: ${total_cost:.2f}")
    
    # Run evaluations
    all_summaries = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for model_config in models:
        api_type = model_config.get("api_type", "openai")
        
        # Check if API key is available
        if api_type == "xai" and not os.environ.get("XAI_API_KEY"):
            print(f"\nSkipping {model_config['name']} - XAI_API_KEY not set")
            continue
        if api_type == "groq" and not os.environ.get("GROQ_API_KEY"):
            print(f"\nSkipping {model_config['name']} - GROQ_API_KEY not set")
            continue
        
        # Output file
        safe_name = model_config["model_id"].replace("/", "_").replace(":", "_")
        output_file = output_dir / f"full_eval_{safe_name}_{timestamp}.json"
        progress_file = output_dir / f"progress_{safe_name}.json"
        
        # Run evaluation
        try:
            result = run_model_evaluation(
                model_config,
                all_problems,
                output_dir,
                resume_file=progress_file
            )
            
            # Calculate summary
            summary = calculate_summary(result["results"])
            result["summary"] = summary
            result["model_name"] = model_config["name"]
            result["timestamp"] = datetime.now().isoformat()
            
            # Save final results
            with open(output_file, "w") as f:
                json.dump(result, f, indent=2)
            
            print(f"\n✅ {model_config['name']}: {summary['passed']}/{summary['total']} ({summary['accuracy']:.1f}%)")
            print(f"   Training: {summary['by_source']['training']['passed']}/{summary['by_source']['training']['total']} ({summary['by_source']['training']['accuracy']:.1f}%)")
            print(f"   Test: {summary['by_source']['test']['passed']}/{summary['by_source']['test']['total']} ({summary['by_source']['test']['accuracy']:.1f}%)")
            print(f"   Saved to: {output_file}")
            
            all_summaries.append({
                "model": model_config["name"],
                "model_id": model_config["model_id"],
                "summary": summary
            })
            
            # Clean up progress file
            if progress_file.exists():
                progress_file.unlink()
                
        except Exception as e:
            print(f"\n❌ Error evaluating {model_config['name']}: {e}")
            import traceback
            traceback.print_exc()
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL RESULTS - ALL MODELS ON 650 PROBLEMS")
    print("="*80)
    
    # Sort by accuracy
    all_summaries.sort(key=lambda x: x["summary"]["accuracy"], reverse=True)
    
    print(f"\n{'Model':<35} {'Total':<8} {'Training':<12} {'Test':<12} {'Overall':<10}")
    print("-"*80)
    
    for s in all_summaries:
        model = s["model"][:32] + "..." if len(s["model"]) > 35 else s["model"]
        total = s["summary"]["accuracy"]
        train = s["summary"]["by_source"]["training"]["accuracy"]
        test = s["summary"]["by_source"]["test"]["accuracy"]
        print(f"{model:<35} {s['summary']['total']:<8} {train:.1f}%{'':<7} {test:.1f}%{'':<7} {total:.1f}%")
    
    # Save comprehensive summary
    summary_file = output_dir / f"comprehensive_summary_{timestamp}.json"
    with open(summary_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total_problems": len(all_problems),
            "training_problems": len(training),
            "test_problems": len(test),
            "results": all_summaries
        }, f, indent=2)
    
    print(f"\n📊 Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
