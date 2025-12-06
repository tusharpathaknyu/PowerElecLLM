#!/usr/bin/env python3
"""
Run SPICE-based evaluation on ALL 650 problems (500 training + 150 test)

This uses ngspice for multi-criteria scoring:
- Vout accuracy (30%)
- Output ripple (25%)
- Efficiency (20%)
- Current ripple (15%)
- Component stress (10%)

Estimated time: ~3-5 hours (ngspice simulation + LLM calls)
Estimated cost: ~$8-10 for API calls
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.evaluate_with_spice import EnhancedLLMEvaluator


def load_all_problems(project_root: Path) -> list:
    """Load all 650 problems"""
    problems = []
    
    # Load training set (500 problems from level_1 to level_4)
    for level in [1, 2, 3, 4]:
        level_dir = project_root / "benchmarks" / f"level_{level}"
        if level_dir.exists():
            for f in sorted(level_dir.glob("problems_*.json")):
                try:
                    with open(f) as fp:
                        data = json.load(fp)
                    for p in data.get("problems", []):
                        p["level"] = level
                        p["source"] = "training"
                        problems.append(p)
                except Exception as e:
                    print(f"Error loading {f}: {e}")
    
    # Load test set (150 problems) - use v2 with ALL topologies
    test_file = project_root / "benchmarks" / "test_set_v2" / "test_problems_v2.json"
    if not test_file.exists():
        # Fallback to original test set
        test_file = project_root / "benchmarks" / "test_set" / "test_problems.json"
    
    if test_file.exists():
        try:
            with open(test_file) as f:
                data = json.load(f)
            for p in data.get("problems", []):
                p["source"] = "test"
                problems.append(p)
            print(f"Loaded {len(data.get('problems', []))} test problems from {test_file.name}")
        except Exception as e:
            print(f"Error loading test set: {e}")
    
    return problems


def run_evaluation(
    model_name: str,
    model_id: str,
    problems: list,
    output_dir: Path,
    use_spice_for_all: bool = True,
    resume: bool = True
):
    """Run SPICE evaluation for a model"""
    
    print(f"\n{'='*70}")
    print(f"Model: {model_name}")
    print(f"Model ID: {model_id}")
    print(f"Problems: {len(problems)}")
    print(f"SPICE for all levels: {use_spice_for_all}")
    print(f"{'='*70}")
    
    # Progress file for resuming
    safe_name = model_id.replace("/", "_").replace(":", "_")
    progress_file = output_dir / f"spice_progress_{safe_name}.json"
    
    # Load existing progress
    results = []
    completed_ids = set()
    
    if resume and progress_file.exists():
        try:
            with open(progress_file) as f:
                existing = json.load(f)
            results = existing.get("results", [])
            completed_ids = {r["problem_id"] for r in results if r.get("problem_id")}
            print(f"Resuming from {len(results)} completed problems")
        except Exception as e:
            print(f"Could not load progress file: {e}")
    
    # Filter remaining problems
    remaining = [p for p in problems if p.get("id") not in completed_ids]
    print(f"Remaining: {len(remaining)}")
    
    if not remaining:
        print("All problems already evaluated!")
        return results
    
    # Create evaluator
    evaluator = EnhancedLLMEvaluator(model_id, use_spice_for_all=use_spice_for_all)
    
    # Track stats
    start_time = time.time()
    total_problems = len(problems)
    
    for i, problem in enumerate(remaining):
        current = len(results) + 1
        problem_id = problem.get("id", f"problem_{i}")
        level = problem.get("level", 1)
        source = problem.get("source", "unknown")
        
        print(f"\n[{current}/{total_problems}] {problem_id} (L{level}, {source})")
        
        # Evaluate
        try:
            result = evaluator.evaluate_problem(problem)
            result["source"] = source
            results.append(result)
            
            # Print result
            if result.get("total_score") is not None:
                score = result.get("total_score", 0)
                vout_sim = result.get("simulated_vout", 0)
                vout_target = result.get("target_vout", 0)
                status = "✓" if result.get("success") else "✗"
                print(f"  {status} Score: {score:.1f}/100, Vout: {vout_sim:.2f}V (target: {vout_target}V)")
                if result.get("ripple_pct"):
                    print(f"     Ripple: {result['ripple_pct']:.2f}%, Efficiency: {result.get('efficiency_pct', 0):.0f}%")
            else:
                status = "✓" if result.get("success") else "✗"
                err = result.get("error_pct", 0)
                print(f"  {status} Simple eval, error: {err:.1f}%")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results.append({
                "problem_id": problem_id,
                "level": level,
                "source": source,
                "success": False,
                "error": str(e)
            })
        
        # Save progress every 5 problems
        if current % 5 == 0:
            with open(progress_file, "w") as f:
                json.dump({"model": model_id, "results": results}, f, indent=2)
            
            # Print running stats
            elapsed = time.time() - start_time
            rate = current / elapsed * 60  # problems per minute
            remaining_time = (total_problems - current) / rate if rate > 0 else 0
            print(f"  [Progress: {current}/{total_problems}, {rate:.1f}/min, ~{remaining_time:.0f}min remaining]")
        
        # Rate limiting
        if "grok" in model_id.lower():
            time.sleep(0.5)
        elif "llama" in model_id.lower():
            time.sleep(1.5)  # Groq rate limit
        else:
            time.sleep(0.3)
    
    return results


def calculate_summary(results: list) -> dict:
    """Calculate comprehensive summary statistics"""
    
    summary = {
        "total": len(results),
        "passed": 0,
        "avg_score": 0,
        "by_level": {},
        "by_source": {
            "training": {"total": 0, "passed": 0, "avg_score": 0},
            "test": {"total": 0, "passed": 0, "avg_score": 0}
        }
    }
    
    scores = []
    by_level_scores = {}
    by_source_scores = {"training": [], "test": []}
    
    for r in results:
        lvl = r.get("level", 1)
        source = r.get("source", "training")
        success = r.get("success", False)
        score = r.get("total_score") or r.get("score", 0)
        
        if lvl not in summary["by_level"]:
            summary["by_level"][lvl] = {"total": 0, "passed": 0, "avg_score": 0}
            by_level_scores[lvl] = []
        
        summary["by_level"][lvl]["total"] += 1
        summary["by_source"][source]["total"] += 1
        
        if success:
            summary["passed"] += 1
            summary["by_level"][lvl]["passed"] += 1
            summary["by_source"][source]["passed"] += 1
        
        if score:
            scores.append(score)
            by_level_scores[lvl].append(score)
            by_source_scores[source].append(score)
    
    # Calculate averages
    if scores:
        summary["avg_score"] = sum(scores) / len(scores)
    
    for lvl, lvl_scores in by_level_scores.items():
        if lvl_scores:
            summary["by_level"][lvl]["avg_score"] = sum(lvl_scores) / len(lvl_scores)
        t = summary["by_level"][lvl]["total"]
        p = summary["by_level"][lvl]["passed"]
        summary["by_level"][lvl]["accuracy"] = p / t * 100 if t > 0 else 0
    
    for src, src_scores in by_source_scores.items():
        if src_scores:
            summary["by_source"][src]["avg_score"] = sum(src_scores) / len(src_scores)
        t = summary["by_source"][src]["total"]
        p = summary["by_source"][src]["passed"]
        summary["by_source"][src]["accuracy"] = p / t * 100 if t > 0 else 0
    
    summary["accuracy"] = summary["passed"] / summary["total"] * 100 if summary["total"] > 0 else 0
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run SPICE evaluation on 650 problems")
    parser.add_argument("--model", type=str, help="Run only this model (by name)")
    parser.add_argument("--use-spice-all", action="store_true", default=True,
                        help="Use SPICE for all levels (not just L3/L4)")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh, don't resume")
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "benchmarks" / "results"
    output_dir.mkdir(exist_ok=True)
    
    # Models configuration
    models = [
        {
            "name": "Fine-tuned GPT-4o",
            "model_id": "ft:gpt-4o-2024-08-06:personal:powerelec:CjLJYOod",
            "api_key_env": "OPENAI_API_KEY"
        },
        {
            "name": "Fine-tuned GPT-4o-mini",
            "model_id": "ft:gpt-4o-mini-2024-07-18:personal:powerelec:CjJntxcS",
            "api_key_env": "OPENAI_API_KEY"
        },
        {
            "name": "GPT-4o (base)",
            "model_id": "gpt-4o",
            "api_key_env": "OPENAI_API_KEY"
        },
        {
            "name": "Grok 4.1 Fast",
            "model_id": "grok-4-1-fast",
            "api_key_env": "XAI_API_KEY"
        },
        {
            "name": "Grok 4.1 Fast Reasoning",
            "model_id": "grok-4-1-fast-reasoning",
            "api_key_env": "XAI_API_KEY"
        },
        {
            "name": "LLaMA 3.3 70B",
            "model_id": "llama-3.3-70b-versatile",
            "api_key_env": "GROQ_API_KEY"
        }
    ]
    
    # Filter models if specified
    if args.model:
        models = [m for m in models if args.model.lower() in m["name"].lower()]
        if not models:
            print(f"No model matching '{args.model}' found")
            return
    
    # Load all problems
    print("Loading 650 problems...")
    all_problems = load_all_problems(project_root)
    
    training = [p for p in all_problems if p.get("source") == "training"]
    test = [p for p in all_problems if p.get("source") == "test"]
    
    print(f"Total: {len(all_problems)}")
    print(f"  Training: {len(training)}")
    print(f"  Test: {len(test)}")
    print(f"  By level: L1={sum(1 for p in all_problems if p.get('level')==1)}, "
          f"L2={sum(1 for p in all_problems if p.get('level')==2)}, "
          f"L3={sum(1 for p in all_problems if p.get('level')==3)}, "
          f"L4={sum(1 for p in all_problems if p.get('level')==4)}")
    
    # Check API keys
    for m in models:
        key = os.environ.get(m["api_key_env"])
        status = "✓" if key else "✗ MISSING"
        print(f"  {m['name']}: {m['api_key_env']} {status}")
    
    # Run evaluations
    all_summaries = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for model in models:
        # Check API key
        if not os.environ.get(model["api_key_env"]):
            print(f"\n⚠️  Skipping {model['name']} - {model['api_key_env']} not set")
            continue
        
        # Run evaluation
        results = run_evaluation(
            model["name"],
            model["model_id"],
            all_problems,
            output_dir,
            use_spice_for_all=args.use_spice_all,
            resume=not args.no_resume
        )
        
        # Calculate summary
        summary = calculate_summary(results)
        
        # Save results
        safe_name = model["model_id"].replace("/", "_").replace(":", "_")
        output_file = output_dir / f"spice_full_{safe_name}_{timestamp}.json"
        
        with open(output_file, "w") as f:
            json.dump({
                "model": model["name"],
                "model_id": model["model_id"],
                "timestamp": datetime.now().isoformat(),
                "summary": summary,
                "results": results
            }, f, indent=2)
        
        print(f"\n✅ {model['name']} Results:")
        print(f"   Overall: {summary['passed']}/{summary['total']} ({summary['accuracy']:.1f}%), Avg Score: {summary['avg_score']:.1f}")
        print(f"   Training: {summary['by_source']['training']['passed']}/{summary['by_source']['training']['total']} "
              f"({summary['by_source']['training']['accuracy']:.1f}%)")
        print(f"   Test: {summary['by_source']['test']['passed']}/{summary['by_source']['test']['total']} "
              f"({summary['by_source']['test']['accuracy']:.1f}%)")
        print(f"   Saved to: {output_file}")
        
        all_summaries.append({
            "model": model["name"],
            "model_id": model["model_id"],
            "summary": summary
        })
        
        # Clean up progress file
        progress_file = output_dir / f"spice_progress_{safe_name}.json"
        if progress_file.exists():
            progress_file.unlink()
    
    # Final comparison
    if len(all_summaries) > 1:
        print("\n" + "="*80)
        print("FINAL RESULTS - SPICE EVALUATION ON 650 PROBLEMS")
        print("="*80)
        
        # Sort by accuracy
        all_summaries.sort(key=lambda x: x["summary"]["accuracy"], reverse=True)
        
        print(f"\n{'Model':<25} {'Total':<8} {'Train':<12} {'Test':<12} {'Avg Score':<10}")
        print("-"*75)
        
        for s in all_summaries:
            model = s["model"][:22] + "..." if len(s["model"]) > 25 else s["model"]
            acc = s["summary"]["accuracy"]
            train = s["summary"]["by_source"]["training"]["accuracy"]
            test = s["summary"]["by_source"]["test"]["accuracy"]
            avg = s["summary"]["avg_score"]
            print(f"{model:<25} {acc:.1f}%{'':<4} {train:.1f}%{'':<7} {test:.1f}%{'':<7} {avg:.1f}")
        
        # Save comprehensive summary
        summary_file = output_dir / f"spice_comprehensive_{timestamp}.json"
        with open(summary_file, "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "total_problems": len(all_problems),
                "training_problems": len(training),
                "test_problems": len(test),
                "use_spice_for_all": args.use_spice_all,
                "results": all_summaries
            }, f, indent=2)
        
        print(f"\n📊 Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
