#!/usr/bin/env python3
"""
Multi-LLM Consensus Solution Generator for Synthetic Power Electronics Problems

Uses GPT-4o, Grok-4.1-Fast-Reasoning, and Gemini-2.0-Flash to generate
consensus solutions. SPICE verification for all solutions.
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

from openai import OpenAI

# Try to import Gemini
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Initialize clients
openai_client = OpenAI()

# API Keys (load from environment variables only - never hardcode!)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
XAI_API_KEY = os.getenv("XAI_API_KEY", "")

if GEMINI_AVAILABLE and GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# Models (GPT-4o, Grok-4.1, Gemini)
MODELS = [
    {"provider": "openai", "model": "gpt-4o", "name": "GPT-4o"},
    {"provider": "xai", "model": "grok-4-1-fast-reasoning", "name": "Grok-4.1-Fast-Reasoning"},
    {"provider": "gemini", "model": "gemini-2.0-flash", "name": "Gemini-2.0-Flash"},
]

DESIGN_PROMPT = """You are a power electronics engineer designing a {topology} converter.

Specifications:
- Input Voltage (Vin): {vin} V
- Output Voltage (Vout): {vout} V  
- Output Current (Iout): {iout} A
- Switching Frequency: {fs} Hz (assume 100kHz if not specified)

Calculate the following parameters. Provide ONLY numerical values:

1. Duty Ratio (D): Calculate based on topology
   - Buck: D = Vout/Vin
   - Boost: D = 1 - Vin/Vout
   - Buck-Boost: D = Vout/(Vin + Vout)

2. Inductor (L): For continuous conduction mode with ripple ≤ 30% of Iout
   - Buck: L = (Vin - Vout) * D / (fs * ΔI)
   - Boost: L = Vin * D / (fs * ΔI)

3. Capacitor (C): For output ripple ≤ 1% of Vout
   - C = Iout * D / (fs * ΔVout)

4. Expected Efficiency: Typical 85-95%

RESPOND IN THIS EXACT FORMAT (numbers only, no units in values):
D={duty_ratio}
L_uH={inductor_in_microhenry}
C_uF={capacitor_in_microfarad}
Efficiency_pct={efficiency_percentage}
"""


def query_openai(prompt: str, model: str = "gpt-4o") -> Tuple[Optional[Dict], str]:
    """Query OpenAI and parse response."""
    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.0
        )
        raw = response.choices[0].message.content
        parsed = parse_design_response(raw)
        return parsed, raw
    except Exception as e:
        return None, str(e)


def query_xai(prompt: str, model: str = "grok-4-1-fast-reasoning") -> Tuple[Optional[Dict], str]:
    """Query xAI Grok and parse response."""
    if not XAI_API_KEY:
        return None, "XAI_API_KEY not set"
    try:
        xai_client = OpenAI(api_key=XAI_API_KEY, base_url="https://api.x.ai/v1")
        response = xai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.0
        )
        raw = response.choices[0].message.content
        parsed = parse_design_response(raw)
        return parsed, raw
    except Exception as e:
        return None, str(e)


def query_gemini(prompt: str, model: str = "gemini-2.0-flash") -> Tuple[Optional[Dict], str]:
    """Query Google Gemini and parse response."""
    if not GEMINI_AVAILABLE or not GOOGLE_API_KEY:
        return None, "Gemini not available"
    try:
        gemini_model = genai.GenerativeModel(model)
        response = gemini_model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(max_output_tokens=300, temperature=0.0)
        )
        raw = response.text
        parsed = parse_design_response(raw)
        return parsed, raw
    except Exception as e:
        return None, str(e)


def parse_design_response(response: str) -> Optional[Dict]:
    """Parse LLM response to extract design parameters."""
    result = {}
    
    # Extract D (duty ratio)
    d_match = re.search(r'D\s*=\s*([\d.]+)', response, re.IGNORECASE)
    if d_match:
        d_val = float(d_match.group(1))
        if d_val > 1:
            d_val = d_val / 100  # Convert percentage
        result['D'] = round(d_val, 4)
    
    # Extract L (inductor in uH)
    l_match = re.search(r'L[_\s]*[uU]?[hH]?\s*=\s*([\d.]+)', response)
    if l_match:
        result['L_uH'] = float(l_match.group(1))
    
    # Extract C (capacitor in uF)
    c_match = re.search(r'C[_\s]*[uU]?[fF]?\s*=\s*([\d.]+)', response)
    if c_match:
        result['C_uF'] = float(c_match.group(1))
    
    # Extract Efficiency
    eff_match = re.search(r'Efficiency[_\s]*pct?\s*=\s*([\d.]+)', response, re.IGNORECASE)
    if eff_match:
        result['Efficiency_pct'] = float(eff_match.group(1))
    
    return result if result.get('D') else None


def get_consensus_solution(solutions: Dict[str, Optional[Dict]]) -> Tuple[Optional[Dict], str, float]:
    """Get consensus solution from multiple LLMs.
    
    Returns: (consensus_solution, method, confidence)
    """
    valid = {k: v for k, v in solutions.items() if v and v.get('D')}
    
    if not valid:
        return None, "no_valid", 0.0
    
    if len(valid) == 1:
        name = list(valid.keys())[0]
        return valid[name], f"single_{name}", 0.33
    
    # Compare D values (rounded to 3 decimals)
    d_values = {k: round(v['D'], 3) for k, v in valid.items()}
    d_counter = Counter(d_values.values())
    most_common = d_counter.most_common()
    
    if len(most_common) == 1:
        # All agree on D
        consensus_d = most_common[0][0]
        # Average other parameters
        consensus = average_solutions([v for v in valid.values()])
        consensus['D'] = consensus_d
        return consensus, "unanimous", 1.0
    
    if most_common[0][1] > 1:
        # Majority agrees
        consensus_d = most_common[0][0]
        matching = [v for k, v in valid.items() if round(v['D'], 3) == consensus_d]
        consensus = average_solutions(matching)
        confidence = most_common[0][1] / len(valid)
        return consensus, "majority", confidence
    
    # All different - use Grok as tiebreaker
    if "Grok-4.1-Fast-Reasoning" in valid:
        return valid["Grok-4.1-Fast-Reasoning"], "grok_tiebreaker", 0.33
    
    # Fallback to first
    return list(valid.values())[0], "first", 0.33


def average_solutions(solutions: List[Dict]) -> Dict:
    """Average multiple solutions."""
    if not solutions:
        return {}
    
    result = {}
    for key in ['D', 'L_uH', 'C_uF', 'Efficiency_pct']:
        values = [s.get(key) for s in solutions if s.get(key) is not None]
        if values:
            result[key] = round(sum(values) / len(values), 4)
    return result


def generate_spice_netlist(topology: str, specs: Dict, solution: Dict) -> str:
    """Generate ngspice netlist for verification."""
    vin = specs.get('vin', 12)
    vout = specs.get('vout', 5)
    iout = specs.get('iout', 1)
    fs = specs.get('fs', 100000)
    
    D = solution.get('D', 0.5)
    L = solution.get('L_uH', 100) * 1e-6  # Convert to H
    C = solution.get('C_uF', 100) * 1e-6  # Convert to F
    
    period = 1 / fs
    ton = D * period
    rload = vout / iout if iout > 0 else 10
    
    if 'buck' in topology.lower():
        netlist = f"""Buck Converter Verification
* Input: {vin}V, Output: {vout}V, D={D}
Vin in 0 DC {vin}
* Ideal switch model
.model sw1 sw(ron=0.01 roff=1e6 vt=0.5 vh=0.1)
S1 in sw ctrl 0 sw1
Vpulse ctrl 0 PULSE(0 1 0 1n 1n {ton:.9f} {period:.9f})
* Freewheeling diode
D1 0 sw DMOD
.model DMOD D(Is=1e-14 Rs=0.01)
* LC filter
L1 sw out {L:.9f}
C1 out 0 {C:.9f}
* Load
Rload out 0 {rload:.3f}
* Simulation
.tran {period/100:.9f} {50*period:.9f}
.control
run
let vout_avg = mean(v(out))
let vout_max = maximum(v(out))
let vout_min = minimum(v(out))
let vout_ripple = vout_max - vout_min
echo "RESULT:vout_avg=$&vout_avg"
echo "RESULT:vout_ripple=$&vout_ripple"
.endc
.end
"""
    elif 'boost' in topology.lower():
        netlist = f"""Boost Converter Verification
* Input: {vin}V, Output: {vout}V, D={D}
Vin in 0 DC {vin}
* Inductor first
L1 in sw {L:.9f}
* Switch
.model sw1 sw(ron=0.01 roff=1e6 vt=0.5 vh=0.1)
S1 sw 0 ctrl 0 sw1
Vpulse ctrl 0 PULSE(0 1 0 1n 1n {ton:.9f} {period:.9f})
* Diode
D1 sw out DMOD
.model DMOD D(Is=1e-14 Rs=0.01)
* Output cap and load
C1 out 0 {C:.9f}
Rload out 0 {rload:.3f}
* Simulation
.tran {period/100:.9f} {50*period:.9f}
.control
run
let vout_avg = mean(v(out))
let vout_ripple = maximum(v(out)) - minimum(v(out))
echo "RESULT:vout_avg=$&vout_avg"
echo "RESULT:vout_ripple=$&vout_ripple"
.endc
.end
"""
    else:
        return None
    
    return netlist


def run_spice_simulation(netlist: str) -> Dict:
    """Run ngspice simulation and parse results."""
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cir', delete=False) as f:
            f.write(netlist)
            netlist_file = f.name
        
        result = subprocess.run(
            ['ngspice', '-b', netlist_file],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        os.unlink(netlist_file)
        
        output = result.stdout + result.stderr
        
        # Parse results
        results = {}
        for line in output.split('\n'):
            if 'RESULT:' in line:
                match = re.search(r'RESULT:(\w+)=([\d.e+-]+)', line)
                if match:
                    results[match.group(1)] = float(match.group(2))
        
        if results:
            results['status'] = 'success'
        else:
            results['status'] = 'parse_error'
            results['output'] = output[-500:]  # Last 500 chars
        
        return results
        
    except subprocess.TimeoutExpired:
        return {'status': 'timeout'}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}


def verify_solution_with_spice(specs: Dict, solution: Dict) -> Dict:
    """Verify solution using SPICE simulation."""
    topology = specs.get('topology', 'buck')
    
    netlist = generate_spice_netlist(topology, specs, solution)
    if not netlist:
        return {'status': 'unsupported_topology', 'topology': topology}
    
    spice_result = run_spice_simulation(netlist)
    
    if spice_result.get('status') == 'success':
        expected_vout = specs.get('vout', 5)
        actual_vout = spice_result.get('vout_avg', 0)
        error_pct = abs(actual_vout - expected_vout) / expected_vout * 100 if expected_vout > 0 else 100
        
        spice_result['expected_vout'] = expected_vout
        spice_result['error_pct'] = round(error_pct, 2)
        spice_result['within_5pct'] = error_pct < 5
        spice_result['within_10pct'] = error_pct < 10
    
    return spice_result


def solve_problem(problem: Dict) -> Dict:
    """Solve a single problem using multi-LLM consensus + SPICE verification."""
    specs = problem.get('specs', {})
    topology = specs.get('topology', 'buck')
    vin = specs.get('vin', 12)
    vout = specs.get('vout', 5)
    iout = specs.get('iout', 1)
    
    prompt = DESIGN_PROMPT.format(
        topology=topology,
        vin=vin,
        vout=vout,
        iout=iout,
        fs=100000
    )
    
    # Query all models
    solutions = {}
    raw_responses = {}
    
    for model_config in MODELS:
        provider = model_config["provider"]
        model = model_config["model"]
        name = model_config["name"]
        
        if provider == "openai":
            parsed, raw = query_openai(prompt, model)
        elif provider == "xai":
            parsed, raw = query_xai(prompt, model)
        elif provider == "gemini":
            parsed, raw = query_gemini(prompt, model)
        else:
            continue
        
        solutions[name] = parsed
        raw_responses[name] = raw
        time.sleep(0.3)
    
    # Get consensus
    consensus, method, confidence = get_consensus_solution(solutions)
    
    # Calculate expected D for reference
    if 'buck' in topology.lower():
        expected_D = vout / vin
    elif 'boost' in topology.lower():
        expected_D = 1 - (vin / vout) if vout > vin else 0.5
    else:
        expected_D = vout / (vin + vout)  # Buck-boost
    
    result = {
        'problem_id': problem['id'],
        'specs': specs,
        'expected_D': round(expected_D, 4),
        'individual_solutions': {k: v for k, v in solutions.items() if v},
        'consensus_solution': consensus,
        'consensus_method': method,
        'consensus_confidence': confidence,
    }
    
    # Verify with SPICE if we have a solution
    if consensus:
        spice_result = verify_solution_with_spice(specs, consensus)
        result['spice_verification'] = spice_result
        result['D_error_pct'] = round(abs(consensus['D'] - expected_D) / expected_D * 100, 2) if expected_D > 0 else 0
    
    return result


def load_all_synthetic_problems() -> List[Dict]:
    """Load all synthetic problems from all levels."""
    all_problems = []
    
    for level in range(1, 6):
        level_dir = Path(f"benchmarks/level_{level}")
        if not level_dir.exists():
            continue
        
        for prob_file in sorted(level_dir.glob("problems_*.json")):
            try:
                with open(prob_file) as f:
                    data = json.load(f)
                    problems = data.get('problems', [])
                    for p in problems:
                        p['level'] = level
                    all_problems.extend(problems)
            except Exception as e:
                print(f"Error loading {prob_file}: {e}")
    
    return all_problems


def main():
    """Main function to generate consensus solutions."""
    print("=" * 70)
    print("Multi-LLM Consensus Solution Generator for Synthetic Problems")
    print("=" * 70)
    print(f"Models: {[m['name'] for m in MODELS]}")
    print(f"Tiebreaker: Grok-4.1-Fast-Reasoning")
    print("=" * 70)
    
    # Load problems
    all_problems = load_all_synthetic_problems()
    print(f"\nTotal synthetic problems: {len(all_problems)}")
    
    # Filter to supported topologies (buck, boost for now)
    supported = [p for p in all_problems 
                 if p.get('specs', {}).get('topology', '').lower() in ['buck', 'boost']]
    print(f"Supported (buck/boost): {len(supported)}")
    
    # Solve problems
    results = []
    
    for i, problem in enumerate(supported):
        print(f"\n[{i+1}/{len(supported)}] Solving {problem['id']}...")
        
        result = solve_problem(problem)
        results.append(result)
        
        # Print summary
        consensus = result.get('consensus_solution', {})
        method = result.get('consensus_method', 'none')
        if consensus:
            d_val = consensus.get('D', 0)
            d_err = result.get('D_error_pct', 0)
            spice = result.get('spice_verification', {})
            spice_status = spice.get('status', 'none')
            spice_err = spice.get('error_pct', 'N/A')
            print(f"  D={d_val:.3f} (err: {d_err:.1f}%) | Method: {method} | SPICE: {spice_status} ({spice_err}%)")
        else:
            print(f"  No consensus solution")
        
        # Rate limiting
        time.sleep(0.5)
        
        # Save progress every 50 problems
        if (i + 1) % 50 == 0:
            save_results(results, "partial")
    
    # Final save
    save_results(results, "final")
    
    # Summary statistics
    print_summary(results)
    
    return results


def save_results(results: List[Dict], suffix: str = ""):
    """Save results to JSON file."""
    output_dir = Path("benchmarks/synthetic")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"consensus_solutions_{suffix}_{timestamp}.json"
    
    output = {
        "generated_at": datetime.now().isoformat(),
        "num_problems": len(results),
        "models": [m['name'] for m in MODELS],
        "solutions": results
    }
    
    with open(output_dir / filename, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {output_dir / filename}")
    
    # Also save to a fixed filename for easy access
    with open(output_dir / "consensus_solutions_latest.json", 'w') as f:
        json.dump(output, f, indent=2)


def print_summary(results: List[Dict]):
    """Print summary statistics."""
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    total = len(results)
    
    # Consensus methods
    methods = Counter(r.get('consensus_method', 'none') for r in results)
    print(f"\nConsensus Methods:")
    for method, count in methods.most_common():
        print(f"  {method}: {count} ({count/total*100:.1f}%)")
    
    # D accuracy
    d_accurate = sum(1 for r in results if r.get('D_error_pct', 100) < 5)
    print(f"\nDuty Ratio Accuracy (<5% error): {d_accurate}/{total} ({d_accurate/total*100:.1f}%)")
    
    # SPICE verification
    spice_success = sum(1 for r in results if r.get('spice_verification', {}).get('status') == 'success')
    spice_accurate = sum(1 for r in results if r.get('spice_verification', {}).get('within_10pct', False))
    print(f"\nSPICE Simulation:")
    print(f"  Successful runs: {spice_success}/{total}")
    print(f"  Within 10% of spec: {spice_accurate}/{total}")


if __name__ == "__main__":
    main()
