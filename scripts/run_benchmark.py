#!/usr/bin/env python3
"""
Benchmark script to run all power electronics tasks and collect statistics.

Usage:
    python scripts/run_benchmark.py --num_runs 3
    python scripts/run_benchmark.py --tasks 1,2,3 --num_runs 5
"""

import argparse
import json
import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent

def load_problem_set():
    """Load problem set from JSON file"""
    problem_file = PROJECT_ROOT / "benchmarks" / "problem_set.json"
    with open(problem_file, 'r') as f:
        return json.load(f)

def run_task(task_id: int, num_retry: int = 3) -> dict:
    """Run a single task and return results"""
    cmd = [
        sys.executable, 
        str(PROJECT_ROOT / "src" / "power_run.py"),
        "--task_id", str(task_id),
        "--num_per_task", "1",
        "--num_of_retry", str(num_retry),
    ]
    
    result = {
        "task_id": task_id,
        "success": False,
        "output_voltage": None,
        "target_voltage": None,
        "error_pct": None,
        "error_msg": None,
    }
    
    try:
        proc = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=300,
            cwd=PROJECT_ROOT
        )
        output = proc.stdout + proc.stderr
        
        # Parse output for results
        if "✅ Generation" in output and "successful" in output:
            result["success"] = True
            
            # Extract output voltage
            import re
            match = re.search(r"Output voltage:\s*([\d.]+)V\s*\(target:\s*([\d.]+)V", output)
            if match:
                result["output_voltage"] = float(match.group(1))
                result["target_voltage"] = float(match.group(2))
                result["error_pct"] = abs(result["output_voltage"] - result["target_voltage"]) / result["target_voltage"] * 100
        else:
            # Extract error message
            if "❌" in output:
                lines = output.split("\n")
                for line in lines:
                    if "❌" in line or "error" in line.lower():
                        result["error_msg"] = line.strip()[:100]
                        break
                        
    except subprocess.TimeoutExpired:
        result["error_msg"] = "Timeout (>300s)"
    except Exception as e:
        result["error_msg"] = str(e)[:100]
    
    return result

def run_benchmark(task_ids: list, num_runs: int = 1, num_retry: int = 3) -> dict:
    """Run benchmark across multiple tasks and runs"""
    problems = load_problem_set()
    problem_map = {p["task_id"]: p for p in problems}
    
    results = defaultdict(list)
    
    total_runs = len(task_ids) * num_runs
    current_run = 0
    
    print(f"\n{'='*70}")
    print(f"PowerElecLLM Benchmark")
    print(f"Tasks: {task_ids}")
    print(f"Runs per task: {num_runs}")
    print(f"Retries per run: {num_retry}")
    print(f"{'='*70}\n")
    
    for task_id in task_ids:
        if task_id not in problem_map:
            print(f"⚠️  Task {task_id} not found in problem set, skipping")
            continue
            
        problem = problem_map[task_id]
        topology = problem["topology"]
        vin = problem["input_voltage"]
        vout = problem["output_voltage"]
        
        print(f"\n📋 Task {task_id}: {topology} {vin}V → {vout}V")
        print("-" * 40)
        
        for run in range(num_runs):
            current_run += 1
            print(f"  Run {run+1}/{num_runs}... ", end="", flush=True)
            
            result = run_task(task_id, num_retry)
            result["topology"] = topology
            result["vin"] = vin
            result["vout"] = vout
            results[task_id].append(result)
            
            if result["success"]:
                print(f"✅ {result['output_voltage']:.2f}V (error: {result['error_pct']:.1f}%)")
            else:
                print(f"❌ {result.get('error_msg', 'Unknown error')[:50]}")
    
    return dict(results)

def print_summary(results: dict):
    """Print summary statistics"""
    print(f"\n{'='*70}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*70}\n")
    
    overall_success = 0
    overall_total = 0
    
    summary_rows = []
    
    for task_id, runs in sorted(results.items()):
        successes = sum(1 for r in runs if r["success"])
        total = len(runs)
        success_rate = successes / total * 100 if total > 0 else 0
        
        overall_success += successes
        overall_total += total
        
        # Calculate average error for successful runs
        successful_errors = [r["error_pct"] for r in runs if r["success"] and r["error_pct"] is not None]
        avg_error = sum(successful_errors) / len(successful_errors) if successful_errors else None
        
        topology = runs[0]["topology"] if runs else "?"
        vin = runs[0]["vin"] if runs else "?"
        vout = runs[0]["vout"] if runs else "?"
        
        summary_rows.append({
            "task_id": task_id,
            "topology": topology,
            "spec": f"{vin}V→{vout}V",
            "success_rate": success_rate,
            "successes": successes,
            "total": total,
            "avg_error": avg_error,
        })
    
    # Print table
    print(f"{'Task':<6} {'Topology':<8} {'Spec':<12} {'Success Rate':<14} {'Avg Error':<10}")
    print("-" * 60)
    
    for row in summary_rows:
        error_str = f"{row['avg_error']:.2f}%" if row['avg_error'] is not None else "N/A"
        print(f"{row['task_id']:<6} {row['topology']:<8} {row['spec']:<12} "
              f"{row['successes']}/{row['total']} ({row['success_rate']:.0f}%){'':>3} {error_str:<10}")
    
    print("-" * 60)
    overall_rate = overall_success / overall_total * 100 if overall_total > 0 else 0
    print(f"{'TOTAL':<6} {'':8} {'':12} {overall_success}/{overall_total} ({overall_rate:.0f}%)")
    
    # Print by topology
    print(f"\n{'='*40}")
    print("BY TOPOLOGY")
    print(f"{'='*40}")
    
    for topology in ["Buck", "Boost"]:
        topo_results = [r for runs in results.values() for r in runs if r.get("topology") == topology]
        if topo_results:
            topo_success = sum(1 for r in topo_results if r["success"])
            topo_total = len(topo_results)
            topo_rate = topo_success / topo_total * 100
            print(f"{topology}: {topo_success}/{topo_total} ({topo_rate:.0f}%)")

def save_results(results: dict, output_file: Path):
    """Save results to JSON file"""
    # Convert to serializable format
    output = {
        "timestamp": datetime.now().isoformat(),
        "results": results,
    }
    
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n📁 Results saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="PowerElecLLM Benchmark Runner")
    parser.add_argument("--tasks", type=str, default=None, 
                       help="Comma-separated task IDs (e.g., '1,2,3'). Default: all tasks")
    parser.add_argument("--num_runs", type=int, default=1,
                       help="Number of runs per task (default: 1)")
    parser.add_argument("--num_retry", type=int, default=3,
                       help="Number of retries per run (default: 3)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output JSON file for results")
    args = parser.parse_args()
    
    # Determine tasks
    if args.tasks:
        task_ids = [int(t.strip()) for t in args.tasks.split(",")]
    else:
        problems = load_problem_set()
        task_ids = [p["task_id"] for p in problems]
    
    # Run benchmark
    results = run_benchmark(task_ids, args.num_runs, args.num_retry)
    
    # Print summary
    print_summary(results)
    
    # Save results
    if args.output:
        output_file = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = PROJECT_ROOT / "benchmarks" / f"results_{timestamp}.json"
    
    save_results(results, output_file)

if __name__ == "__main__":
    main()
