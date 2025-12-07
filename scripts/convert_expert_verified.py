#!/usr/bin/env python3
"""
Convert expert-verified problems to the standard benchmark format for evaluation.
These problems have physics-derived ground truth, not LLM-generated solutions.
"""

import json
import os
from pathlib import Path

def convert_physics_verified_to_benchmark():
    """Convert physics-verified problems to evaluation format."""
    
    input_file = Path("benchmarks/expert_verified/physics_verified_problems.json")
    output_dir = Path("benchmarks/expert_verified_eval")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(input_file) as f:
        data = json.load(f)
    
    problems = []
    for p in data["problems"]:
        # Map difficulty to level
        difficulty_map = {"easy": 1, "medium": 2, "hard": 3, "expert": 4}
        level = difficulty_map.get(p.get("difficulty", "medium"), 2)
        
        # Extract specs
        specs = p.get("specs", {})
        gt = p.get("ground_truth", {})
        
        problem = {
            "id": p["id"],
            "topology": p["topology"],
            "level": level,
            "source": "physics_verified",
            "specs": {
                "vin": specs.get("vin") or specs.get("vin_nom"),
                "vout": abs(specs.get("vout") or specs.get("vout_magnitude", 0)),
                "iout": specs.get("iout"),
                "pout": specs.get("pout"),
                "frequency": specs.get("frequency"),
                "ripple_voltage_pct": specs.get("ripple_voltage_percent"),
                "ripple_current_pct": specs.get("ripple_current_percent"),
            },
            "ground_truth": {
                "duty_cycle": gt.get("duty_cycle") or gt.get("duty_cycle_at_vin_nom"),
                "inductance_uH": gt.get("inductance_min_uH") or gt.get("output_inductance_uH") or gt.get("inductance_L1_uH"),
                "capacitance_uF": gt.get("capacitance_min_uF") or gt.get("output_capacitance_uF"),
                "formula_duty_cycle": gt.get("duty_cycle_formula"),
                "formula_inductance": gt.get("inductance_formula"),
                "formula_capacitance": gt.get("capacitance_formula"),
            },
            "verification": p.get("verification", {}),
            "problem_statement": p.get("problem_statement"),
        }
        
        # Remove None values
        problem["specs"] = {k: v for k, v in problem["specs"].items() if v is not None}
        problem["ground_truth"] = {k: v for k, v in problem["ground_truth"].items() if v is not None}
        
        problems.append(problem)
    
    # Group by topology
    by_topology = {}
    for p in problems:
        topo = p["topology"]
        if topo not in by_topology:
            by_topology[topo] = []
        by_topology[topo].append(p)
    
    # Save grouped files
    for topo, probs in by_topology.items():
        output_file = output_dir / f"problems_{topo}.json"
        output_data = {
            "metadata": {
                "source": "physics_verified",
                "topology": topo,
                "count": len(probs),
                "verification": "analytical_formulas"
            },
            "problems": probs
        }
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"Saved {len(probs)} {topo} problems to {output_file}")
    
    # Save all problems in one file
    all_file = output_dir / "all_physics_verified.json"
    with open(all_file, "w") as f:
        json.dump({
            "metadata": data["metadata"],
            "problems": problems
        }, f, indent=2)
    print(f"Saved all {len(problems)} problems to {all_file}")
    
    return problems


def convert_gate_style_to_benchmark():
    """Convert GATE-style problems to evaluation format."""
    
    input_file = Path("benchmarks/expert_verified/gate_style_problems.json")
    output_dir = Path("benchmarks/expert_verified_eval")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(input_file) as f:
        data = json.load(f)
    
    problems = []
    for p in data["problems"]:
        # Extract topology from topic or problem statement
        topic = p.get("topic", "").lower()
        if "buck-boost" in topic or "buck_boost" in topic:
            topology = "buck_boost"
        elif "buck" in topic:
            topology = "buck"
        elif "boost" in topic:
            topology = "boost"
        elif "flyback" in topic:
            topology = "flyback"
        elif "forward" in topic:
            topology = "forward"
        elif "sepic" in topic:
            topology = "sepic"
        elif "cuk" in topic:
            topology = "cuk"
        elif "half-bridge" in topic or "half_bridge" in topic:
            topology = "half_bridge"
        elif "full-bridge" in topic or "full_bridge" in topic:
            topology = "full_bridge"
        else:
            topology = "unknown"
        
        # Map marks to difficulty
        marks = p.get("marks", 1)
        level = 1 if marks == 1 else 2
        
        problem = {
            "id": p["id"],
            "topology": topology,
            "level": level,
            "source": "gate_style",
            "type": p.get("type", "numerical"),
            "marks": marks,
            "topic": p.get("topic"),
            "problem_statement": p.get("problem_statement"),
            "ground_truth": {
                "answer": p.get("answer"),
                "solution": p.get("solution"),
                "tolerance": p.get("tolerance")
            }
        }
        problems.append(problem)
    
    # Save to file
    output_file = output_dir / "gate_style_problems.json"
    with open(output_file, "w") as f:
        json.dump({
            "metadata": data["metadata"],
            "problems": problems
        }, f, indent=2)
    print(f"Saved {len(problems)} GATE-style problems to {output_file}")
    
    return problems


def create_summary():
    """Create a summary of all expert-verified problems."""
    
    output_dir = Path("benchmarks/expert_verified_eval")
    
    # Load all problems
    physics_file = output_dir / "all_physics_verified.json"
    gate_file = output_dir / "gate_style_problems.json"
    
    physics_count = 0
    gate_count = 0
    
    if physics_file.exists():
        with open(physics_file) as f:
            data = json.load(f)
            physics_count = len(data.get("problems", []))
    
    if gate_file.exists():
        with open(gate_file) as f:
            data = json.load(f)
            gate_count = len(data.get("problems", []))
    
    summary = {
        "total_expert_verified_problems": physics_count + gate_count,
        "physics_verified_design_problems": physics_count,
        "gate_style_numerical_problems": gate_count,
        "verification_methods": [
            "Analytical formulas from textbooks",
            "SPICE simulation validation",
            "Closed-form solutions"
        ],
        "topologies_covered": [
            "buck", "boost", "buck_boost", "flyback", 
            "sepic", "forward", "half_bridge", "full_bridge", "cuk"
        ],
        "key_advantage": "Ground truth derived from physics, not LLM-generated"
    }
    
    summary_file = output_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*60)
    print("EXPERT-VERIFIED BENCHMARK SUMMARY")
    print("="*60)
    print(f"Physics-verified design problems: {physics_count}")
    print(f"GATE-style numerical problems: {gate_count}")
    print(f"TOTAL: {physics_count + gate_count} problems")
    print("="*60)
    
    return summary


if __name__ == "__main__":
    print("Converting expert-verified problems to benchmark format...\n")
    
    convert_physics_verified_to_benchmark()
    print()
    convert_gate_style_to_benchmark()
    print()
    create_summary()
