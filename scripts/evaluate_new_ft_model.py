#!/usr/bin/env python3
"""
Evaluate the new fine-tuned model (2000 examples) on the full 650-problem benchmark.
Compare with previous results.
"""

import json
import sys
import os
import time
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI

# Import SPICE evaluator
try:
    from src.spice_evaluator import PowerConverterEvaluator
    SPICE_AVAILABLE = True
except ImportError:
    SPICE_AVAILABLE = False
    print("Warning: SPICE evaluator not available")

# Model configurations
NEW_FT_MODEL = "ft:gpt-4o-mini-2024-07-18:personal:powerelec-v2:Cjv8xPFg"  # 2000 examples
OLD_FT_4O = "ft:gpt-4o-2024-08-06:personal:powerelec:CjLJYOod"  # Previous best
OLD_FT_MINI = "ft:gpt-4o-mini-2024-07-18:personal:powerelec:CjJntxcS"  # Previous mini

def load_problems(filepath):
    """Load problems from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def get_model_response(client, model, problem, use_cot=False):
    """Get model response for a problem."""
    
    if use_cot:
        system_prompt = """You are an expert power electronics engineer. Solve problems step-by-step.

Key Formulas:
- Buck: D = Vout/Vin, L = Vout(1-D)/(f*ΔIL), C = ΔIL/(8*f*ΔVout)
- Boost: D = 1 - Vin/Vout, L = Vin*D/(f*ΔIL), C = Iout*D/(f*ΔVout)
- Buck-Boost: D = Vout/(Vin+Vout), L = Vin*D/(f*ΔIL)

Always show your calculations clearly, then provide the final answer as JSON."""
    else:
        system_prompt = "You are a power electronics expert. Provide component values in JSON format."
    
    user_prompt = f"""Design a {problem['topology']} converter:
- Input voltage: {problem['input_voltage']}V
- Output voltage: {problem['output_voltage']}V  
- Output current: {problem['output_current']}A
- Switching frequency: {problem['switching_frequency']}Hz
- Ripple requirements: {problem.get('voltage_ripple', 1)}% voltage, {problem.get('current_ripple', 20)}% current

Provide component values as JSON: {{"D": duty_cycle, "L": inductance_H, "C": capacitance_F, "R_load": load_ohms}}"""
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"API Error: {e}")
        return None

def parse_response(response):
    """Extract component values from model response."""
    if not response:
        return None
    
    import re
    
    # Try to find JSON in response
    json_match = re.search(r'\{[^}]+\}', response.replace('\n', ' '))
    if json_match:
        try:
            values = json.loads(json_match.group())
            # Normalize keys
            result = {}
            for key in ['D', 'd', 'duty_cycle', 'duty']:
                if key in values:
                    result['D'] = float(values[key])
                    break
            for key in ['L', 'l', 'inductance', 'inductor']:
                if key in values:
                    result['L'] = float(values[key])
                    break
            for key in ['C', 'c', 'capacitance', 'capacitor']:
                if key in values:
                    result['C'] = float(values[key])
                    break
            for key in ['R_load', 'r_load', 'R', 'load', 'resistance']:
                if key in values:
                    result['R_load'] = float(values[key])
                    break
            return result if 'D' in result else None
        except:
            pass
    
    return None

def validate_with_spice(problem, components):
    """Validate design using SPICE simulation."""
    if not SPICE_AVAILABLE or not components:
        return None
    
    try:
        evaluator = PowerConverterEvaluator()
        result = evaluator.evaluate_design(
            topology=problem['topology'],
            components=components,
            specs={
                'input_voltage': problem['input_voltage'],
                'output_voltage': problem['output_voltage'],
                'output_current': problem['output_current'],
                'switching_frequency': problem['switching_frequency'],
                'voltage_ripple': problem.get('voltage_ripple', 1),
                'current_ripple': problem.get('current_ripple', 20)
            }
        )
        return result
    except Exception as e:
        print(f"SPICE error: {e}")
        return None

def run_evaluation(client, model, problems, model_name, use_cot=False, max_problems=None):
    """Run evaluation on problems."""
    
    results = {
        'model': model,
        'model_name': model_name,
        'use_cot': use_cot,
        'timestamp': datetime.now().isoformat(),
        'total_problems': 0,
        'correct': 0,
        'incorrect': 0,
        'errors': 0,
        'by_topology': {},
        'details': []
    }
    
    problems_to_test = problems[:max_problems] if max_problems else problems
    results['total_problems'] = len(problems_to_test)
    
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"CoT Prompting: {'Yes' if use_cot else 'No'}")
    print(f"Problems: {len(problems_to_test)}")
    print(f"{'='*60}")
    
    for i, problem in enumerate(problems_to_test):
        topology = problem['topology']
        
        if topology not in results['by_topology']:
            results['by_topology'][topology] = {'correct': 0, 'total': 0}
        results['by_topology'][topology]['total'] += 1
        
        # Get model response
        response = get_model_response(client, model, problem, use_cot)
        components = parse_response(response)
        
        # Validate
        is_correct = False
        validation = None
        
        if components:
            validation = validate_with_spice(problem, components)
            if validation and validation.get('overall_pass', False):
                is_correct = True
        
        # Record result
        if components is None:
            results['errors'] += 1
            status = "❌ PARSE ERROR"
        elif is_correct:
            results['correct'] += 1
            results['by_topology'][topology]['correct'] += 1
            status = "✅ CORRECT"
        else:
            results['incorrect'] += 1
            status = "❌ INCORRECT"
        
        results['details'].append({
            'problem_id': i,
            'topology': topology,
            'components': components,
            'is_correct': is_correct,
            'validation': validation
        })
        
        # Progress update
        if (i + 1) % 50 == 0 or i == len(problems_to_test) - 1:
            accuracy = results['correct'] / (i + 1) * 100
            print(f"Progress: {i+1}/{len(problems_to_test)} | "
                  f"Accuracy: {accuracy:.1f}% | "
                  f"Last: {topology} {status}")
        
        # Rate limiting
        time.sleep(0.3)
    
    # Calculate final accuracy
    results['accuracy'] = results['correct'] / results['total_problems'] * 100
    
    return results

def main():
    client = OpenAI()
    
    # Load problems
    problems_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                  'benchmarks', 'problem_set.json')
    
    with open(problems_file, 'r') as f:
        data = json.load(f)
    
    # Flatten all problems
    all_problems = []
    for level, level_problems in data.items():
        all_problems.extend(level_problems)
    
    print(f"Loaded {len(all_problems)} problems")
    
    # Run evaluations
    all_results = []
    
    # Test new fine-tuned model (2000 examples)
    print("\n" + "="*70)
    print("EVALUATING NEW FINE-TUNED MODEL (2000 examples)")
    print("="*70)
    
    # Basic prompting
    results_new_ft = run_evaluation(
        client, NEW_FT_MODEL, all_problems,
        "FT GPT-4o-mini v2 (2000 examples) - Basic",
        use_cot=False
    )
    all_results.append(results_new_ft)
    
    # With CoT prompting
    results_new_ft_cot = run_evaluation(
        client, NEW_FT_MODEL, all_problems,
        "FT GPT-4o-mini v2 (2000 examples) - CoT",
        use_cot=True
    )
    all_results.append(results_new_ft_cot)
    
    # Save results
    output_file = os.path.join(os.path.dirname(__file__), 'new_ft_model_results.json')
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    
    print("\n📊 NEW FINE-TUNED MODEL (2000 examples) RESULTS:")
    print("-" * 50)
    
    for r in all_results:
        print(f"\n{r['model_name']}:")
        print(f"  Overall Accuracy: {r['accuracy']:.1f}%")
        print(f"  Correct: {r['correct']}/{r['total_problems']}")
        print(f"  Parse Errors: {r['errors']}")
        
        print(f"\n  By Topology:")
        for topo, stats in sorted(r['by_topology'].items()):
            topo_acc = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
            print(f"    {topo}: {stats['correct']}/{stats['total']} ({topo_acc:.1f}%)")
    
    # Comparison with previous models
    print("\n" + "="*70)
    print("COMPARISON WITH PREVIOUS RESULTS")
    print("="*70)
    print("""
Previous Results (650 problems):
- FT GPT-4o (400 examples):      25.0%
- FT GPT-4o-mini (400 examples): 23.8%
- GPT-4o (base):                 21.4%
- Grok 4.1 Fast Reasoning:       19.1%
- LLaMA 3.3 70B:                  2.3%
""")
    
    for r in all_results:
        improvement = r['accuracy'] - 23.8  # vs old mini
        print(f"NEW {r['model_name']}: {r['accuracy']:.1f}% ({'+' if improvement > 0 else ''}{improvement:.1f}%)")
    
    print(f"\nResults saved to: {output_file}")
    
    return all_results

if __name__ == "__main__":
    main()
