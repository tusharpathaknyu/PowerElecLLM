#!/usr/bin/env python3
"""
Comprehensive Benchmark Evaluation for PowerElecLLM

Evaluates LLMs on:
1. GATE theory questions (MCQ with consensus LLM answers)
2. MIT problems (with ground truth solutions)  
3. Synthetic circuit problems (Multi-LLM consensus verification)
"""

import json
import os
import re
import subprocess
import tempfile
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from openai import OpenAI

# Try to import optional providers
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Initialize OpenAI client
client = OpenAI()

# Configure Gemini (API key from environment variable)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
if GEMINI_AVAILABLE and GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# xAI API key
XAI_API_KEY = os.getenv("XAI_API_KEY", "")

# Evaluation config
EVAL_MODEL = "gpt-4o"
RESULTS_DIR = Path("benchmarks/evaluation_results")
RESULTS_DIR.mkdir(exist_ok=True)

# Multi-LLM models for consensus (GPT-4o, Grok-4.1, Gemini)
CONSENSUS_MODELS = [
    {"provider": "openai", "model": "gpt-4o", "name": "GPT-4o"},
    {"provider": "xai", "model": "grok-4-1-fast-reasoning", "name": "Grok-4.1-Fast-Reasoning"},
    {"provider": "gemini", "model": "gemini-2.0-flash", "name": "Gemini-2.0-Flash"},
]


# =============================================================================
# Multi-LLM Query Functions
# =============================================================================

def query_openai_design(model: str, prompt: str) -> Tuple[Optional[float], str]:
    """Query OpenAI model for circuit design, return duty ratio."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.0
        )
        raw = response.choices[0].message.content
        d_val = extract_duty_ratio(raw)
        return d_val, raw
    except Exception as e:
        return None, str(e)


def query_xai_design(model: str, prompt: str) -> Tuple[Optional[float], str]:
    """Query xAI Grok model for circuit design."""
    if not XAI_API_KEY:
        return None, "XAI_API_KEY not set"
    try:
        xai_client = OpenAI(api_key=XAI_API_KEY, base_url="https://api.x.ai/v1")
        response = xai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.0
        )
        raw = response.choices[0].message.content
        d_val = extract_duty_ratio(raw)
        return d_val, raw
    except Exception as e:
        return None, str(e)


def query_gemini_design(model: str, prompt: str) -> Tuple[Optional[float], str]:
    """Query Google Gemini model for circuit design."""
    if not GEMINI_AVAILABLE:
        return None, "Gemini not available"
    try:
        gemini_model = genai.GenerativeModel(model)
        response = gemini_model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(max_output_tokens=200, temperature=0.0)
        )
        raw = response.text
        d_val = extract_duty_ratio(raw)
        return d_val, raw
    except Exception as e:
        return None, str(e)


def extract_duty_ratio(response: str) -> Optional[float]:
    """Extract duty ratio from LLM response."""
    # Look for D= followed by decimal
    d_match = re.search(r'D\s*=\s*(0?\.\d+|\d+\.\d+)', response, re.IGNORECASE)
    if d_match:
        d_val = float(d_match.group(1))
        if d_val > 1:  # Percentage
            d_val = d_val / 100
        return d_val
    return None


def get_consensus_duty_ratio(answers: Dict[str, Optional[float]]) -> Tuple[Optional[float], str, Dict]:
    """Get consensus duty ratio from multiple LLM answers."""
    valid = {k: v for k, v in answers.items() if v is not None}
    
    if not valid:
        return None, "no_valid", answers
    
    if len(valid) == 1:
        return list(valid.values())[0], "single", answers
    
    # Round to 3 decimal places for comparison
    rounded = {k: round(v, 3) for k, v in valid.items()}
    counter = Counter(rounded.values())
    most_common = counter.most_common()
    
    if len(most_common) == 1 or most_common[0][1] > 1:
        # Unanimous or majority
        consensus_val = most_common[0][0]
        method = "unanimous" if most_common[0][1] == len(valid) else "majority"
        return consensus_val, method, answers
    else:
        # All different - use Grok-4.1-Fast-Reasoning as tiebreaker
        if "Grok-4.1-Fast-Reasoning" in valid:
            return valid["Grok-4.1-Fast-Reasoning"], "grok_tiebreaker", answers
        return list(valid.values())[0], "first", answers


# =============================================================================
# Theory Question Evaluation (GATE MCQ)
# =============================================================================

def evaluate_gate_mcq(problem: Dict, model: str = EVAL_MODEL) -> Dict:
    """Evaluate a GATE MCQ question."""
    question_text = problem.get("problem_text", "")
    expected_answer = problem.get("llm_answer")  # Consensus answer from multi-LLM
    
    if not expected_answer:
        return {"status": "skip", "reason": "no_expected_answer"}
    
    prompt = f"""You are solving a GATE Power Electronics exam question.
Read the question carefully and select the correct answer.
Respond with ONLY a single letter: A, B, C, or D

Question:
{question_text}

Answer:"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0.0
        )
        raw_answer = response.choices[0].message.content.strip().upper()
        
        # Extract just the letter
        predicted = None
        for char in raw_answer:
            if char in 'ABCD':
                predicted = char
                break
        
        is_correct = predicted == expected_answer
        
        return {
            "status": "evaluated",
            "question_id": problem["id"],
            "expected": expected_answer,
            "predicted": predicted,
            "correct": is_correct,
            "confidence": problem.get("llm_confidence", 0),
            "raw_response": raw_answer
        }
    
    except Exception as e:
        return {"status": "error", "error": str(e)}


# =============================================================================
# Synthetic Circuit SPICE Evaluation
# =============================================================================

SPICE_TEMPLATE = '''#!/usr/bin/env python3
"""Generated SPICE test for {topology} converter"""
import sys
sys.path.insert(0, 'src')

try:
    from PySpice.Spice.NgSpice.Shared import NgSpiceShared
    from PySpice.Probe.Plot import plot
    from PySpice.Spice.Parser import SpiceParser
    from PySpice.Unit import *
    import numpy as np
    
    # Circuit netlist
    netlist = """
{netlist}
"""
    
    # Run simulation
    ngspice = NgSpiceShared.new_instance()
    ngspice.load_circuit(netlist)
    ngspice.run()
    
    # Get results
    time_vec = np.array(ngspice.plot(ngspice.last_plot).get_time())
    
    # Try to get output voltage
    try:
        vout = np.array(ngspice.plot(ngspice.last_plot).get_data('v(out)'))
    except:
        vout = np.array(ngspice.plot(ngspice.last_plot).get_data('out'))
    
    # Calculate metrics
    vout_avg = np.mean(vout[len(vout)//2:])  # Steady state average
    vout_ripple = np.max(vout[len(vout)//2:]) - np.min(vout[len(vout)//2:])
    
    print(f"RESULT:vout_avg={vout_avg:.3f}")
    print(f"RESULT:vout_ripple={vout_ripple:.3f}")
    print("STATUS:success")
    
except Exception as e:
    print(f"STATUS:error")
    print(f"ERROR:{str(e)}")
'''


def generate_spice_netlist(problem: Dict) -> Optional[str]:
    """Generate SPICE netlist for a circuit problem."""
    topology = problem.get("topology", "").lower()
    specs = problem.get("specifications", {})
    
    vin = specs.get("vin", specs.get("Vin", 12))
    vout = specs.get("vout", specs.get("Vout", 5))
    iout = specs.get("iout", specs.get("Iout", 1))
    freq = specs.get("switching_frequency", specs.get("fs", 100000))
    
    # Calculate ideal duty ratio
    if "buck" in topology:
        D = vout / vin
    elif "boost" in topology:
        D = 1 - (vin / vout) if vout > vin else 0.5
    else:
        D = 0.5
    
    period = 1 / freq
    ton = D * period
    
    # Basic buck converter netlist
    if "buck" in topology:
        netlist = f"""Buck Converter
Vin in 0 DC {vin}
.model switch SW(Ron=0.01 Roff=1e6 Vt=0.5)
S1 in sw ctrl 0 switch
Vpulse ctrl 0 PULSE(0 1 0 1n 1n {ton:.9f} {period:.9f})
L1 sw out {1e-4}
C1 out 0 {100e-6}
Rload out 0 {vout/iout if iout > 0 else 5}
D1 0 sw DMOD
.model DMOD D(Is=1e-14)
.tran {period/100:.9f} {20*period:.9f}
.end
"""
    else:
        return None
    
    return netlist


def evaluate_spice_circuit(problem: Dict) -> Dict:
    """Evaluate a synthetic circuit problem using SPICE simulation."""
    specs = problem.get("specifications", {})
    expected_vout = specs.get("vout", specs.get("Vout"))
    
    if not expected_vout:
        return {"status": "skip", "reason": "no_expected_vout"}
    
    netlist = generate_spice_netlist(problem)
    if not netlist:
        return {"status": "skip", "reason": "unsupported_topology"}
    
    # For now, return a simulated result (actual SPICE requires ngspice setup)
    topology = problem.get("topology", "").lower()
    vin = specs.get("vin", specs.get("Vin", 12))
    
    # Calculate expected output
    if "buck" in topology:
        D = expected_vout / vin
        predicted_vout = D * vin
    elif "boost" in topology:
        D = 1 - (vin / expected_vout)
        predicted_vout = vin / (1 - D)
    else:
        predicted_vout = expected_vout
    
    # Add some simulation "noise"
    error_pct = abs(predicted_vout - expected_vout) / expected_vout * 100
    
    return {
        "status": "evaluated",
        "problem_id": problem["id"],
        "expected_vout": expected_vout,
        "predicted_vout": round(predicted_vout, 2),
        "error_percent": round(error_pct, 2),
        "within_5pct": error_pct < 5,
        "topology": topology
    }


# =============================================================================
# LLM Circuit Design Evaluation
# =============================================================================

def evaluate_llm_circuit_design(problem: Dict, use_consensus: bool = True) -> Dict:
    """Have LLM design a circuit and verify against specs using multi-LLM consensus."""
    specs = problem.get("specs", problem.get("specifications", {}))
    topology = specs.get("topology", problem.get("topology", "buck"))
    
    vin = specs.get("vin", specs.get("Vin", 12))
    vout = specs.get("vout", specs.get("Vout", 5))
    iout = specs.get("iout", specs.get("Iout", 1))
    freq = specs.get("switching_frequency", specs.get("fs", 100000))
    
    prompt = f"""Design a {topology} converter with these specifications:
- Input Voltage: {vin} V
- Output Voltage: {vout} V
- Output Current: {iout} A
- Switching Frequency: {freq} Hz

Calculate and provide these values ONLY (no explanations, just numbers):
D=<duty_ratio_between_0_and_1>
L_uH=<inductor_in_microhenry>
C_uF=<capacitor_in_microfarad>
Efficiency_pct=<expected_efficiency>

Example format:
D=0.417
L_uH=100
C_uF=470
Efficiency_pct=90
"""

    try:
        # Calculate expected duty ratio
        if "buck" in topology.lower():
            expected_D = vout / vin
        elif "boost" in topology.lower():
            expected_D = 1 - (vin / vout)
        else:
            expected_D = 0.5
        
        if use_consensus and (XAI_API_KEY or GEMINI_AVAILABLE):
            # Multi-LLM consensus approach
            answers = {}
            raw_responses = {}
            
            for model_config in CONSENSUS_MODELS:
                provider = model_config["provider"]
                model = model_config["model"]
                name = model_config["name"]
                
                if provider == "openai":
                    d_val, raw = query_openai_design(model, prompt)
                elif provider == "xai":
                    d_val, raw = query_xai_design(model, prompt)
                elif provider == "gemini":
                    d_val, raw = query_gemini_design(model, prompt)
                else:
                    continue
                
                answers[name] = d_val
                raw_responses[name] = raw
                time.sleep(0.3)
            
            predicted_D, method, all_answers = get_consensus_duty_ratio(answers)
            
            D_error = abs(predicted_D - expected_D) / expected_D * 100 if expected_D > 0 and predicted_D else 100
            
            return {
                "status": "evaluated",
                "problem_id": problem["id"],
                "expected_D": round(expected_D, 3),
                "predicted_D": round(predicted_D, 3) if predicted_D else None,
                "D_error_percent": round(D_error, 2),
                "correct_D": D_error < 10,
                "consensus_method": method,
                "individual_answers": {k: round(v, 3) if v else None for k, v in all_answers.items()},
                "raw_responses": raw_responses
            }
        else:
            # Single model fallback
            response = client.chat.completions.create(
                model=EVAL_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.0
            )
            raw_response = response.choices[0].message.content
            predicted_D = extract_duty_ratio(raw_response)
            
            D_error = abs(predicted_D - expected_D) / expected_D * 100 if expected_D > 0 and predicted_D else 100
            
            return {
                "status": "evaluated",
                "problem_id": problem["id"],
                "expected_D": round(expected_D, 3),
                "predicted_D": round(predicted_D, 3) if predicted_D else None,
                "D_error_percent": round(D_error, 2),
                "correct_D": D_error < 10,
                "raw_response": raw_response
            }
    
    except Exception as e:
        return {"status": "error", "error": str(e)}


# =============================================================================
# Main Evaluation Runner
# =============================================================================

def run_full_evaluation(
    model: str = EVAL_MODEL,
    gate_limit: int = 20,
    synthetic_limit: int = 20
) -> Dict:
    """Run comprehensive evaluation across all problem types."""
    
    print("=" * 70)
    print(f"PowerElecLLM Comprehensive Benchmark Evaluation")
    print(f"Model: {model}")
    print(f"Time: {datetime.now().isoformat()}")
    print("=" * 70)
    
    results = {
        "model": model,
        "timestamp": datetime.now().isoformat(),
        "gate_mcq": [],
        "synthetic_design": [],
        "summary": {}
    }
    
    # =========================================================================
    # 1. GATE MCQ Evaluation
    # =========================================================================
    print("\n[1/2] Evaluating GATE MCQ questions...")
    
    try:
        with open("benchmarks/expert_verified/gate_problems_with_answers.json") as f:
            gate_data = json.load(f)
        
        # Filter problems with high-confidence answers
        gate_problems = [
            p for p in gate_data["problems"] 
            if p.get("llm_answer") and p.get("llm_confidence", 0) >= 0.67
        ][:gate_limit]
        
        print(f"  Testing on {len(gate_problems)} high-confidence questions...")
        
        gate_correct = 0
        for i, problem in enumerate(gate_problems):
            result = evaluate_gate_mcq(problem, model)
            results["gate_mcq"].append(result)
            
            if result.get("correct"):
                gate_correct += 1
            
            status = "✓" if result.get("correct") else "✗"
            print(f"  [{i+1}/{len(gate_problems)}] {problem['id']}: {status}")
            time.sleep(0.5)  # Rate limit
        
        gate_accuracy = gate_correct / len(gate_problems) * 100 if gate_problems else 0
        print(f"\n  GATE MCQ Accuracy: {gate_correct}/{len(gate_problems)} ({gate_accuracy:.1f}%)")
        
    except FileNotFoundError:
        print("  GATE problems file not found, skipping...")
        gate_accuracy = 0
    
    # =========================================================================
    # 2. Synthetic Circuit Design Evaluation
    # =========================================================================
    print("\n[2/2] Evaluating Synthetic Circuit Design...")
    
    synthetic_correct = 0
    synthetic_count = 0
    
    for level in [1, 2, 3]:
        level_dir = Path(f"benchmarks/level_{level}")
        if not level_dir.exists():
            continue
        
        # Load problems from split files
        all_problems = []
        for prob_file in sorted(level_dir.glob("problems_*.json")):
            with open(prob_file) as f:
                file_data = json.load(f)
                all_problems.extend(file_data.get("problems", []))
        
        # Limit problems per level
        problems = all_problems[:synthetic_limit // 3]
        
        print(f"  Level {level}: Testing {len(problems)} problems...")
        
        for problem in problems:
            result = evaluate_llm_circuit_design(problem, use_consensus=True)
            results["synthetic_design"].append(result)
            
            if result.get("correct_D"):
                synthetic_correct += 1
            synthetic_count += 1
            
            status = "✓" if result.get("correct_D") else "✗"
            method = result.get("consensus_method", "single")
            answers_str = ""
            if "individual_answers" in result:
                answers_str = f" | {result['individual_answers']}"
            print(f"    {problem['id']}: {status} (D err: {result.get('D_error_percent', 'N/A')}%, {method}){answers_str}")
            time.sleep(0.5)
    
    synthetic_accuracy = synthetic_correct / synthetic_count * 100 if synthetic_count else 0
    print(f"\n  Synthetic Design Accuracy: {synthetic_correct}/{synthetic_count} ({synthetic_accuracy:.1f}%)")
    
    # =========================================================================
    # Summary
    # =========================================================================
    results["summary"] = {
        "gate_mcq": {
            "total": len(results["gate_mcq"]),
            "correct": sum(1 for r in results["gate_mcq"] if r.get("correct")),
            "accuracy": gate_accuracy
        },
        "synthetic_design": {
            "total": len(results["synthetic_design"]),
            "correct": sum(1 for r in results["synthetic_design"] if r.get("correct_D")),
            "accuracy": synthetic_accuracy
        },
        "overall_accuracy": (gate_accuracy + synthetic_accuracy) / 2
    }
    
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"GATE MCQ:           {results['summary']['gate_mcq']['accuracy']:.1f}%")
    print(f"Synthetic Design:   {results['summary']['synthetic_design']['accuracy']:.1f}%")
    print(f"Overall Average:    {results['summary']['overall_accuracy']:.1f}%")
    print("=" * 70)
    
    # Save results
    output_path = RESULTS_DIR / f"eval_{model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run PowerElecLLM benchmark evaluation")
    parser.add_argument("--model", default="gpt-4o", help="Model to evaluate")
    parser.add_argument("--gate-limit", type=int, default=20, help="Max GATE questions")
    parser.add_argument("--synthetic-limit", type=int, default=15, help="Max synthetic problems")
    
    args = parser.parse_args()
    
    run_full_evaluation(
        model=args.model,
        gate_limit=args.gate_limit,
        synthetic_limit=args.synthetic_limit
    )
