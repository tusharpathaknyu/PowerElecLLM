#!/usr/bin/env python3
"""
Create high-quality fine-tuning dataset for LLaMA on power electronics.
Generates proper solutions with step-by-step calculations.
Format: Together AI / OpenAI compatible JSONL
"""

import json
import os
from pathlib import Path
from typing import Dict, List

SYSTEM_PROMPT = """You are an expert power electronics engineer specializing in DC-DC converter design. When given a design problem:

1. Identify the topology (buck, boost, buck-boost, etc.)
2. Calculate the duty cycle using the correct formula
3. Determine component values (L, C) for the given specifications
4. Verify your answer

Always show calculations step-by-step. Use proper formulas:
- Buck: D = Vout/Vin
- Boost: D = 1 - Vin/Vout  
- Buck-Boost: D = Vout/(Vin + Vout)"""


def calculate_solution(specs: Dict) -> str:
    """Generate a detailed solution with calculations."""
    topology = specs.get('topology', 'buck')
    vin = specs.get('vin', 24)
    vout = specs.get('vout', 12)
    iout = specs.get('iout', 2)
    fsw = specs.get('fsw', 100000)
    ripple_pct = specs.get('vout_ripple_pct', 5)
    power = specs.get('power', vout * iout)
    
    # Calculate duty cycle based on topology
    if topology == 'buck':
        D = vout / vin
        formula = f"D = Vout/Vin = {vout}/{vin} = {D:.4f}"
        vout_formula = "Vout = Vin × D"
    elif topology == 'boost':
        D = 1 - vin / vout
        formula = f"D = 1 - Vin/Vout = 1 - {vin}/{vout} = {D:.4f}"
        vout_formula = "Vout = Vin / (1-D)"
    elif topology == 'buck_boost' or topology == 'buck-boost':
        D = abs(vout) / (vin + abs(vout))
        formula = f"D = |Vout|/(Vin + |Vout|) = {abs(vout)}/({vin} + {abs(vout)}) = {D:.4f}"
        vout_formula = "Vout = -Vin × D / (1-D)"
    elif topology == 'sepic':
        D = vout / (vin + vout)
        formula = f"D = Vout/(Vin + Vout) = {vout}/({vin} + {vout}) = {D:.4f}"
        vout_formula = "Vout = Vin × D / (1-D)"
    elif topology == 'cuk':
        D = abs(vout) / (vin + abs(vout))
        formula = f"D = |Vout|/(Vin + |Vout|) = {abs(vout)}/({vin} + {abs(vout)}) = {D:.4f}"
        vout_formula = "Vout = -Vin × D / (1-D)"
    elif topology in ['flyback', 'forward', 'half_bridge', 'full_bridge', 'push_pull']:
        n = specs.get('n', 1.0)  # turns ratio
        if topology == 'flyback':
            D = vout / (vout + vin * n)
            formula = f"D = Vout/(Vout + Vin×n) = {vout}/({vout} + {vin}×{n}) = {D:.4f}"
        elif topology == 'forward':
            D = vout / (vin * n)
            formula = f"D = Vout/(Vin×n) = {vout}/({vin}×{n}) = {D:.4f}"
        else:  # half_bridge, full_bridge, push_pull
            D = vout / (vin * n)
            formula = f"D = Vout/(Vin×n) = {vout}/({vin}×{n}) = {D:.4f}"
        vout_formula = f"Vout = Vin × D × n (n={n})"
    else:
        D = vout / vin
        formula = f"D = Vout/Vin = {vout}/{vin} = {D:.4f}"
        vout_formula = "Vout = Vin × D"
    
    # Clamp duty cycle
    D = max(0.05, min(0.95, D))
    
    # Calculate inductance for CCM
    R_load = vout / iout if iout > 0 else 10
    if topology == 'buck':
        L_min = (vout * (1 - D)) / (2 * fsw * iout * 0.3)  # 30% ripple
    elif topology == 'boost':
        L_min = (vin * D) / (2 * fsw * iout * 0.3)
    else:
        L_min = (vin * D) / (2 * fsw * iout * 0.3) if iout > 0 else 100e-6
    
    L = max(L_min, 10e-6)  # Minimum 10µH
    
    # Calculate capacitance for ripple
    delta_vout = vout * ripple_pct / 100
    C = iout * D / (fsw * delta_vout) if delta_vout > 0 else 100e-6
    C = max(C, 10e-6)  # Minimum 10µF
    
    solution = f"""**Analysis:**
- Input Voltage: Vin = {vin}V
- Output Voltage: Vout = {vout}V
- Output Current: Iout = {iout}A
- Output Power: P = {power}W
- Switching Frequency: fsw = {fsw/1000:.0f}kHz

**Topology Selection:** {topology.upper()} converter
- {vout_formula}

**Duty Cycle Calculation:**
{formula}
D = {D:.4f} ({D*100:.1f}%)

**Inductor Calculation (for CCM with 30% current ripple):**
L_min = Vin × D / (2 × fsw × ΔIL)
L = {L*1e6:.1f}µH (selected)

**Output Capacitor Calculation (for {ripple_pct}% voltage ripple):**
C = Iout × D / (fsw × ΔVout)
C = {C*1e6:.1f}µF (selected)

**Load Resistance:**
R_load = Vout / Iout = {vout}/{iout} = {R_load:.2f}Ω

**Final Answer:**
- Topology: {topology}
- Duty Cycle: D = {D:.4f} ({D*100:.1f}%)
- Inductance: L = {L*1e6:.1f}µH
- Capacitance: C = {C*1e6:.1f}µF
- Output Voltage: Vout = {vout}V"""
    
    return solution


def create_training_example(problem: Dict) -> Dict:
    """Create a single training example."""
    prompt = problem.get('prompt', '')
    specs = problem.get('specs', {})
    
    # Create detailed solution
    solution = calculate_solution(specs)
    
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": solution}
        ]
    }


def main():
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "benchmarks" / "finetune"
    output_dir.mkdir(exist_ok=True)
    
    # Collect all problems from levels 1-4
    all_examples = []
    
    for level in [1, 2, 3, 4]:
        level_dir = project_root / "benchmarks" / f"level_{level}"
        if not level_dir.exists():
            continue
            
        for json_file in sorted(level_dir.glob("problems_*.json")):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                
                for problem in data.get('problems', []):
                    example = create_training_example(problem)
                    all_examples.append(example)
                    
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
    
    print(f"Generated {len(all_examples)} training examples")
    
    # Split into train/val (90/10)
    split_idx = int(len(all_examples) * 0.9)
    train_examples = all_examples[:split_idx]
    val_examples = all_examples[split_idx:]
    
    # Save in JSONL format
    train_file = output_dir / "llama_train.jsonl"
    val_file = output_dir / "llama_val.jsonl"
    
    with open(train_file, 'w') as f:
        for ex in train_examples:
            f.write(json.dumps(ex) + '\n')
    
    with open(val_file, 'w') as f:
        for ex in val_examples:
            f.write(json.dumps(ex) + '\n')
    
    print(f"Saved {len(train_examples)} training examples to {train_file}")
    print(f"Saved {len(val_examples)} validation examples to {val_file}")
    
    # Also create Together AI format (same as OpenAI)
    together_file = output_dir / "powerelec_together.jsonl"
    with open(together_file, 'w') as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + '\n')
    
    print(f"Saved {len(all_examples)} total examples to {together_file}")
    
    # Show sample
    print("\n" + "="*60)
    print("SAMPLE TRAINING EXAMPLE:")
    print("="*60)
    sample = all_examples[0]
    print(f"User: {sample['messages'][1]['content'][:100]}...")
    print(f"\nAssistant: {sample['messages'][2]['content'][:500]}...")


if __name__ == "__main__":
    main()
