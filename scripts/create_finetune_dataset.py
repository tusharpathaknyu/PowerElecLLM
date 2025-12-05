#!/usr/bin/env python3
"""
Create fine-tuning dataset from PowerElecBench solutions.
Generates JSONL format compatible with OpenAI fine-tuning API.
"""

import json
import os
import glob
from typing import List, Dict, Any

SYSTEM_PROMPT = """You are an expert power electronics engineer. When given a power electronics design problem, analyze the requirements and provide:

1. **Analysis**: Brief analysis of the problem requirements
2. **Topology Selection**: Choose appropriate converter topology with justification
3. **Component Values**: Calculate all component values with formulas
4. **Final Answer**: Output voltage (Vout) clearly stated

Always show your calculations step-by-step and use proper engineering notation."""

def solution_dict_to_text(sol_dict: Dict) -> str:
    """Convert structured solution dict to readable text format."""
    if isinstance(sol_dict, str):
        return sol_dict
    
    lines = []
    
    # Topology
    if 'topology' in sol_dict:
        lines.append(f"**Topology Selection:** {sol_dict['topology'].upper()} converter")
    
    # Specifications
    specs = []
    if 'vin' in sol_dict:
        specs.append(f"Vin = {sol_dict['vin']}V")
    if 'vout' in sol_dict:
        specs.append(f"Vout = {sol_dict['vout']}V")
    if 'iout' in sol_dict:
        specs.append(f"Iout = {sol_dict['iout']}A")
    if 'power' in sol_dict:
        specs.append(f"Power = {sol_dict['power']}W")
    if 'f_sw' in sol_dict:
        specs.append(f"Switching Frequency = {sol_dict['f_sw']/1000:.0f}kHz")
    if specs:
        lines.append(f"\n**Specifications:** {', '.join(specs)}")
    
    # Duty Cycle
    if 'duty_cycle' in sol_dict:
        dc = sol_dict['duty_cycle']
        if isinstance(dc, (int, float)):
            lines.append(f"\n**Duty Cycle:** D = {dc:.3f}")
        else:
            lines.append(f"\n**Duty Cycle:** D = {dc}")
    
    # Design Equations
    if 'design_equations' in sol_dict:
        lines.append("\n**Design Calculations:**")
        for name, eq in sol_dict['design_equations'].items():
            lines.append(f"- {name}: {eq}")
    
    # Components
    if 'components' in sol_dict:
        lines.append("\n**Component Values:**")
        for comp, info in sol_dict['components'].items():
            if isinstance(info, dict):
                value = info.get('value', 'N/A')
                unit = info.get('unit', '')
                # Format scientific notation
                if isinstance(value, float) and value < 0.001:
                    value_str = f"{value*1e6:.1f}µ{unit}" if value < 1e-3 else f"{value*1e3:.1f}m{unit}"
                else:
                    value_str = f"{value}{unit}"
                lines.append(f"- {comp}: {value_str}")
            else:
                lines.append(f"- {comp}: {info}")
    
    # Expected Results
    if 'expected_results' in sol_dict:
        lines.append("\n**Expected Results:**")
        results = sol_dict['expected_results']
        if 'vout_actual' in results:
            lines.append(f"- Output Voltage: {results['vout_actual']}V")
        if 'ripple_mv' in results:
            lines.append(f"- Output Ripple: {results['ripple_mv']}mV")
        if 'efficiency_est' in results:
            lines.append(f"- Estimated Efficiency: {results['efficiency_est']*100:.0f}%")
    
    # Final Answer
    if 'vout' in sol_dict:
        lines.append(f"\n**Final Answer:** Vout = {sol_dict['vout']}V")
    
    return '\n'.join(lines)

def extract_problem_solution_pairs(level: int) -> List[Dict[str, Any]]:
    """Extract problem-solution pairs from benchmark files."""
    pairs = []
    
    problems_dir = f'benchmarks/level_{level}'
    solutions_dir = f'benchmarks/level_{level}'
    
    # Load all problems
    problem_files = sorted(glob.glob(f'{problems_dir}/problems_*.json'))
    solution_files = sorted(glob.glob(f'{solutions_dir}/solutions_*.json'))
    
    # Create lookup dict for solutions
    solutions_lookup = {}
    for sol_file in solution_files:
        with open(sol_file, 'r') as f:
            data = json.load(f)
            for sol in data.get('solutions', []):
                sol_id = sol.get('id', sol.get('problem_id', ''))
                solutions_lookup[sol_id] = sol
    
    # Process each problem
    for prob_file in problem_files:
        with open(prob_file, 'r') as f:
            data = json.load(f)
            
        for problem in data.get('problems', []):
            prob_id = problem.get('id', '')
            
            # Find matching solution
            solution = solutions_lookup.get(prob_id)
            if not solution:
                continue
            
            # Extract problem statement
            prob_statement = problem.get('problem_statement', '')
            if not prob_statement:
                # Try to construct from specs
                specs = problem.get('specs', {})
                prob_statement = json.dumps(specs, indent=2)
            
            # Extract solution - handle both dict and string formats
            sol_raw = solution.get('solution', '')
            if isinstance(sol_raw, dict):
                sol_text = solution_dict_to_text(sol_raw)
                vout = sol_raw.get('vout', '')
            else:
                sol_text = sol_raw
                # Get Vout from other locations
                vout = solution.get('specs', {}).get('vout', '')
                if not vout:
                    vout = solution.get('specifications', {}).get('V_out', '')
            
            if not sol_text:
                sol_text = solution.get('detailed_solution', '')
                if not sol_text:
                    sol_text = solution.get('analysis', '')
            
            if prob_statement and sol_text:
                pairs.append({
                    'id': prob_id,
                    'level': level,
                    'problem': prob_statement,
                    'solution': sol_text,
                    'vout': vout
                })
    
    return pairs

def create_openai_finetune_format(pairs: List[Dict]) -> List[Dict]:
    """Convert to OpenAI fine-tuning JSONL format."""
    finetune_data = []
    
    for pair in pairs:
        entry = {
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user", 
                    "content": f"Design Problem:\n\n{pair['problem']}"
                },
                {
                    "role": "assistant",
                    "content": pair['solution']
                }
            ]
        }
        finetune_data.append(entry)
    
    return finetune_data

def create_alpaca_format(pairs: List[Dict]) -> List[Dict]:
    """Convert to Alpaca format for open-source fine-tuning."""
    alpaca_data = []
    
    for pair in pairs:
        entry = {
            "instruction": "You are a power electronics engineer. Solve the following design problem, showing your analysis, topology selection, component calculations, and final output voltage (Vout).",
            "input": pair['problem'],
            "output": pair['solution'],
            "metadata": {
                "id": pair['id'],
                "level": pair['level'],
                "vout": pair['vout']
            }
        }
        alpaca_data.append(entry)
    
    return alpaca_data

def main():
    print("🔧 Creating Fine-Tuning Dataset from PowerElecBench")
    print("=" * 60)
    
    os.makedirs('benchmarks/finetune', exist_ok=True)
    
    all_pairs = []
    
    # Extract from all levels
    for level in [1, 2, 3, 4]:
        pairs = extract_problem_solution_pairs(level)
        print(f"Level {level}: {len(pairs)} problem-solution pairs")
        all_pairs.extend(pairs)
    
    print(f"\nTotal pairs: {len(all_pairs)}")
    
    # Create OpenAI format
    openai_data = create_openai_finetune_format(all_pairs)
    with open('benchmarks/finetune/powerelec_openai.jsonl', 'w') as f:
        for entry in openai_data:
            f.write(json.dumps(entry) + '\n')
    print(f"✅ OpenAI format: benchmarks/finetune/powerelec_openai.jsonl ({len(openai_data)} examples)")
    
    # Create Alpaca format
    alpaca_data = create_alpaca_format(all_pairs)
    with open('benchmarks/finetune/powerelec_alpaca.json', 'w') as f:
        json.dump(alpaca_data, f, indent=2)
    print(f"✅ Alpaca format: benchmarks/finetune/powerelec_alpaca.json ({len(alpaca_data)} examples)")
    
    # Create train/val split (90/10)
    import random
    random.seed(42)
    random.shuffle(openai_data)
    
    split_idx = int(len(openai_data) * 0.9)
    train_data = openai_data[:split_idx]
    val_data = openai_data[split_idx:]
    
    with open('benchmarks/finetune/train.jsonl', 'w') as f:
        for entry in train_data:
            f.write(json.dumps(entry) + '\n')
    
    with open('benchmarks/finetune/val.jsonl', 'w') as f:
        for entry in val_data:
            f.write(json.dumps(entry) + '\n')
    
    print(f"✅ Train split: benchmarks/finetune/train.jsonl ({len(train_data)} examples)")
    print(f"✅ Val split: benchmarks/finetune/val.jsonl ({len(val_data)} examples)")
    
    # Statistics by level
    print("\n" + "=" * 60)
    print("Dataset Statistics:")
    print("=" * 60)
    level_counts = {}
    for pair in all_pairs:
        level_counts[pair['level']] = level_counts.get(pair['level'], 0) + 1
    
    for level in sorted(level_counts.keys()):
        print(f"  Level {level}: {level_counts[level]} examples")
    
    # Token estimation
    total_chars = sum(len(p['problem']) + len(p['solution']) for p in all_pairs)
    est_tokens = total_chars // 4  # Rough estimate
    print(f"\nEstimated tokens: ~{est_tokens:,}")
    print(f"Fine-tuning cost estimate (GPT-4o-mini): ~${est_tokens * 0.003 / 1000:.2f}")
    
    print("\n" + "=" * 60)
    print("📁 All files saved to: benchmarks/finetune/")
    print("=" * 60)

if __name__ == "__main__":
    main()
