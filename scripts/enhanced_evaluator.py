#!/usr/bin/env python3
"""
Enhanced LLM Evaluator with:
1. Chain-of-Thought (CoT) Prompting
2. Few-Shot Examples
3. Tool-Augmented LLM (Calculator)

This improves accuracy by guiding the model through proper reasoning.
"""

import json
import os
import re
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from openai import OpenAI

PROJECT_ROOT = Path(__file__).parent.parent


# ============================================================================
# CHAIN-OF-THOUGHT PROMPTS
# ============================================================================

SYSTEM_PROMPT_COT = """You are an expert power electronics engineer designing DC-DC converters.

CRITICAL: You must follow these steps IN ORDER and show ALL calculations:

## STEP 1: IDENTIFY TOPOLOGY
First, identify the converter type from the problem.

## STEP 2: WRITE THE DUTY CYCLE FORMULA
Use the CORRECT formula (this is critical!):
- Buck converter: D = Vout / Vin
- Boost converter: D = 1 - (Vin / Vout)
- Buck-Boost converter: D = |Vout| / (Vin + |Vout|)
- SEPIC converter: D = Vout / (Vin + Vout)
- Cuk converter: D = |Vout| / (Vin + |Vout|)
- Flyback converter: D = Vout / (Vout + Vin × n)
- Forward converter: D = Vout / (Vin × n)

## STEP 3: SUBSTITUTE VALUES AND CALCULATE
Show the arithmetic step by step. Use a calculator if needed.

## STEP 4: VERIFY
Check that D is between 0.05 and 0.95. If not, reconsider your design.

## STEP 5: CALCULATE COMPONENTS
- Inductor: L = Vin × D × (1-D) / (2 × fsw × ΔIL)
- Capacitor: C = Iout × D / (fsw × ΔVout)
- Load: R = Vout / Iout

## FINAL ANSWER FORMAT
Always end with a JSON block:
```json
{
  "topology": "buck",
  "D": 0.417,
  "L": 47e-6,
  "C": 100e-6,
  "R": 2.5,
  "Vout_expected": 5.0
}
```"""


FEW_SHOT_EXAMPLES = [
    {
        "role": "user",
        "content": "Design a buck converter: 12V input to 5V output, 2A load current"
    },
    {
        "role": "assistant", 
        "content": """## STEP 1: IDENTIFY TOPOLOGY
This is a **BUCK** converter (step-down: 12V → 5V).

## STEP 2: WRITE THE DUTY CYCLE FORMULA
For Buck converter: **D = Vout / Vin**

## STEP 3: SUBSTITUTE VALUES AND CALCULATE
D = Vout / Vin
D = 5V / 12V
D = 0.4167

Let me verify: 5 ÷ 12 = 0.41666...
**D = 0.417 (41.7%)**

## STEP 4: VERIFY
✓ D = 0.417 is between 0.05 and 0.95 - valid!
✓ Expected Vout = Vin × D = 12 × 0.417 = 5.0V ✓

## STEP 5: CALCULATE COMPONENTS
Given: fsw = 100kHz (assumed), ΔIL = 30% of Iout = 0.6A

**Inductor:**
L = Vin × D × (1-D) / (2 × fsw × ΔIL)
L = 12 × 0.417 × 0.583 / (2 × 100000 × 0.6)
L = 2.916 / 120000
L = 24.3µH → Use **L = 22µH** (standard value)

**Capacitor (for 5% ripple = 0.25V):**
C = Iout × D / (fsw × ΔVout)
C = 2 × 0.417 / (100000 × 0.25)
C = 33.3µF → Use **C = 47µF**

**Load Resistance:**
R = Vout / Iout = 5V / 2A = **2.5Ω**

```json
{
  "topology": "buck",
  "D": 0.417,
  "L": 22e-6,
  "C": 47e-6,
  "R": 2.5,
  "Vout_expected": 5.0
}
```"""
    },
    {
        "role": "user",
        "content": "Design a boost converter: 5V input to 12V output, 1A load"
    },
    {
        "role": "assistant",
        "content": """## STEP 1: IDENTIFY TOPOLOGY
This is a **BOOST** converter (step-up: 5V → 12V).

## STEP 2: WRITE THE DUTY CYCLE FORMULA
For Boost converter: **D = 1 - (Vin / Vout)**

⚠️ NOT D = Vout/Vin! That's for Buck only!

## STEP 3: SUBSTITUTE VALUES AND CALCULATE
D = 1 - (Vin / Vout)
D = 1 - (5V / 12V)
D = 1 - 0.4167
D = 0.5833

Let me verify: 5 ÷ 12 = 0.4167, then 1 - 0.4167 = 0.5833
**D = 0.583 (58.3%)**

## STEP 4: VERIFY
✓ D = 0.583 is between 0.05 and 0.95 - valid!
✓ Expected Vout = Vin / (1-D) = 5 / (1-0.583) = 5 / 0.417 = 12.0V ✓

## STEP 5: CALCULATE COMPONENTS
Given: fsw = 100kHz, Iout = 1A

**Input current (boost draws more from input):**
Iin = Iout / (1-D) = 1 / 0.417 = 2.4A

**Inductor (30% ripple = 0.72A):**
L = Vin × D / (fsw × ΔIL)
L = 5 × 0.583 / (100000 × 0.72)
L = 40.5µH → Use **L = 47µH**

**Capacitor (for 5% ripple = 0.6V):**
C = Iout × D / (fsw × ΔVout)
C = 1 × 0.583 / (100000 × 0.6)
C = 9.7µF → Use **C = 22µF**

**Load Resistance:**
R = Vout / Iout = 12V / 1A = **12Ω**

```json
{
  "topology": "boost",
  "D": 0.583,
  "L": 47e-6,
  "C": 22e-6,
  "R": 12.0,
  "Vout_expected": 12.0
}
```"""
    },
    {
        "role": "user",
        "content": "Design a buck-boost converter: 12V input to -15V output, 1A load"
    },
    {
        "role": "assistant",
        "content": """## STEP 1: IDENTIFY TOPOLOGY
This is a **BUCK-BOOST** converter (inverts polarity: 12V → -15V).

## STEP 2: WRITE THE DUTY CYCLE FORMULA
For Buck-Boost: **D = |Vout| / (Vin + |Vout|)**

⚠️ Use absolute value of Vout since output is negative!

## STEP 3: SUBSTITUTE VALUES AND CALCULATE
D = |Vout| / (Vin + |Vout|)
D = |-15| / (12 + |-15|)
D = 15 / (12 + 15)
D = 15 / 27
D = 0.5556

Let me verify: 15 ÷ 27 = 0.5555...
**D = 0.556 (55.6%)**

## STEP 4: VERIFY
✓ D = 0.556 is between 0.05 and 0.95 - valid!
✓ Expected Vout = -Vin × D / (1-D) = -12 × 0.556 / 0.444 = -15.0V ✓

## STEP 5: CALCULATE COMPONENTS
Given: fsw = 100kHz, Iout = 1A

**Inductor:**
L = Vin × D / (fsw × ΔIL)
L = 12 × 0.556 / (100000 × 0.3)
L = 222µH → Use **L = 220µH**

**Capacitor:**
C = Iout × D / (fsw × ΔVout)
C = 1 × 0.556 / (100000 × 0.75)
C = 7.4µF → Use **C = 10µF**

**Load Resistance:**
R = |Vout| / Iout = 15V / 1A = **15Ω**

```json
{
  "topology": "buck_boost",
  "D": 0.556,
  "L": 220e-6,
  "C": 10e-6,
  "R": 15.0,
  "Vout_expected": -15.0
}
```"""
    }
]


# ============================================================================
# TOOL-AUGMENTED PROMPTS (with calculator)
# ============================================================================

SYSTEM_PROMPT_TOOLS = """You are an expert power electronics engineer. You have access to a CALCULATOR tool.

When you need to perform calculations, use this format:
<calc>expression</calc>

The calculator will return the result. Example:
<calc>5/12</calc> → 0.4167
<calc>1 - 5/12</calc> → 0.5833

ALWAYS use the calculator for duty cycle calculations to avoid errors!

## DUTY CYCLE FORMULAS (use calculator!):
- Buck: D = Vout / Vin → <calc>Vout/Vin</calc>
- Boost: D = 1 - Vin/Vout → <calc>1 - Vin/Vout</calc>
- Buck-Boost: D = |Vout| / (Vin + |Vout|) → <calc>|Vout|/(Vin + |Vout|)</calc>

After calculations, provide final answer as JSON:
```json
{"topology": "...", "D": ..., "L": ..., "C": ..., "R": ...}
```"""


def calculator_tool(expression: str) -> float:
    """Safe calculator for power electronics expressions."""
    # Clean the expression
    expr = expression.strip()
    
    # Replace common patterns
    expr = expr.replace('×', '*').replace('÷', '/').replace('^', '**')
    expr = re.sub(r'\|(-?\d+\.?\d*)\|', r'abs(\1)', expr)  # |x| → abs(x)
    
    # Safe evaluation with math functions
    allowed_names = {
        'abs': abs, 'sqrt': math.sqrt, 'pow': pow,
        'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
        'log': math.log, 'log10': math.log10, 'exp': math.exp,
        'pi': math.pi, 'e': math.e
    }
    
    try:
        result = eval(expr, {"__builtins__": {}}, allowed_names)
        return round(float(result), 6)
    except Exception as e:
        return float('nan')


def process_calculator_calls(text: str) -> str:
    """Process <calc>...</calc> tags and replace with results."""
    pattern = r'<calc>([^<]+)</calc>'
    
    def replace_calc(match):
        expr = match.group(1)
        result = calculator_tool(expr)
        return f"<calc>{expr}</calc> = **{result}**"
    
    return re.sub(pattern, replace_calc, text)


# ============================================================================
# ENHANCED EVALUATOR CLASS
# ============================================================================

class EnhancedEvaluator:
    """Evaluator with CoT, Few-Shot, and Tool-Augmented capabilities."""
    
    def __init__(self, model: str = "gpt-4o", mode: str = "cot"):
        """
        Args:
            model: Model to use (gpt-4o, gpt-4o-mini, etc.)
            mode: 'basic', 'cot', 'few_shot', 'tools', or 'all'
        """
        self.client = OpenAI()
        self.model = model
        self.mode = mode
        
    def build_messages(self, prompt: str) -> List[Dict]:
        """Build message list based on mode."""
        messages = []
        
        if self.mode == "basic":
            messages = [
                {"role": "system", "content": "You are a power electronics expert. Return a JSON with: topology, D (duty cycle), L, C, R."},
                {"role": "user", "content": prompt}
            ]
            
        elif self.mode == "cot":
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT_COT},
                {"role": "user", "content": prompt}
            ]
            
        elif self.mode == "few_shot":
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT_COT},
                *FEW_SHOT_EXAMPLES,
                {"role": "user", "content": prompt}
            ]
            
        elif self.mode == "tools":
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT_TOOLS},
                {"role": "user", "content": prompt}
            ]
            
        elif self.mode == "all":
            # Combine few-shot with tool instructions
            combined_system = SYSTEM_PROMPT_COT + "\n\n" + """
You can also use a calculator: <calc>expression</calc>
Example: <calc>5/12</calc> returns 0.4167

USE THE CALCULATOR for all duty cycle calculations!"""
            messages = [
                {"role": "system", "content": combined_system},
                *FEW_SHOT_EXAMPLES,
                {"role": "user", "content": prompt}
            ]
            
        return messages
    
    def evaluate(self, prompt: str) -> Dict:
        """Evaluate a single problem."""
        messages = self.build_messages(prompt)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content
            
            # Process calculator calls if present
            if "<calc>" in content:
                content = process_calculator_calls(content)
            
            # Extract JSON from response
            params = self.extract_params(content)
            
            return {
                "success": True,
                "raw_response": content,
                "params": params
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "params": None
            }
    
    def extract_params(self, text: str) -> Optional[Dict]:
        """Extract parameters from response."""
        # Try to find JSON block
        json_match = re.search(r'```json\s*(\{[^`]+\})\s*```', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except:
                pass
        
        # Try to find bare JSON
        json_match = re.search(r'\{[^{}]*"D"[^{}]*\}', text)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except:
                pass
        
        # Extract individual values
        params = {}
        
        # Duty cycle
        d_match = re.search(r'["\']?D["\']?\s*[:=]\s*([0-9.]+)', text)
        if d_match:
            params['D'] = float(d_match.group(1))
        
        # Topology
        for topo in ['buck_boost', 'buck', 'boost', 'sepic', 'cuk', 'flyback', 'forward', 'half_bridge', 'full_bridge', 'push_pull']:
            if topo in text.lower():
                params['topology'] = topo
                break
        
        return params if params else None


# ============================================================================
# COMPARISON RUNNER
# ============================================================================

def run_comparison(num_problems: int = 20):
    """Compare different prompting strategies."""
    
    # Load test problems from level_1 (has proper topology labels)
    problems = []
    for file_num in ["01_10", "11_20", "21_30"]:
        test_file = PROJECT_ROOT / "benchmarks" / "level_1" / f"problems_{file_num}.json"
        try:
            with open(test_file) as f:
                test_data = json.load(f)
                problems.extend(test_data.get("problems", []))
        except:
            pass
    
    problems = problems[:num_problems]
    
    modes = ["basic", "cot", "few_shot", "all"]
    results = {mode: {"correct": 0, "total": 0, "details": []} for mode in modes}
    
    print("=" * 70)
    print("PROMPTING STRATEGY COMPARISON")
    print("=" * 70)
    
    for i, problem in enumerate(problems):
        prompt = problem.get("prompt", "")
        specs = problem.get("specs", {})
        target_vout = specs.get("vout", 0)
        topology = specs.get("topology", "buck")
        vin = specs.get("vin", 12)
        
        # Calculate expected duty cycle
        if topology == "buck":
            expected_D = target_vout / vin
        elif topology == "boost":
            expected_D = 1 - vin / target_vout
        elif topology in ["buck_boost", "cuk"]:
            expected_D = abs(target_vout) / (vin + abs(target_vout))
        elif topology == "sepic":
            expected_D = target_vout / (vin + target_vout)
        else:
            expected_D = 0.5
        
        print(f"\n[{i+1}/{num_problems}] {problem.get('id', 'unknown')}")
        print(f"    Target: {topology} {vin}V→{target_vout}V, Expected D={expected_D:.4f}")
        
        for mode in modes:
            evaluator = EnhancedEvaluator(model="gpt-4o-mini", mode=mode)
            result = evaluator.evaluate(prompt)
            
            results[mode]["total"] += 1
            
            if result["success"] and result["params"]:
                D = result["params"].get("D", 0)
                error = abs(D - expected_D) / expected_D * 100 if expected_D > 0 else 100
                
                if error < 5:  # Within 5% of correct D
                    results[mode]["correct"] += 1
                    status = "✓"
                else:
                    status = "✗"
                
                print(f"    {mode:10s}: D={D:.4f} (error={error:.1f}%) {status}")
                results[mode]["details"].append({"id": problem.get("id"), "D": D, "error": error})
            else:
                print(f"    {mode:10s}: FAILED")
                results[mode]["details"].append({"id": problem.get("id"), "error": "failed"})
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Mode':<15} {'Correct':>10} {'Total':>10} {'Accuracy':>12}")
    print("-" * 50)
    for mode in modes:
        r = results[mode]
        acc = r["correct"] / r["total"] * 100 if r["total"] > 0 else 0
        print(f"{mode:<15} {r['correct']:>10} {r['total']:>10} {acc:>11.1f}%")
    
    # Save results
    output_file = PROJECT_ROOT / "benchmarks" / "results" / "prompting_comparison.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    return results


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced LLM Evaluator")
    parser.add_argument("--mode", choices=["basic", "cot", "few_shot", "tools", "all", "compare"], 
                        default="compare", help="Evaluation mode")
    parser.add_argument("--num", type=int, default=20, help="Number of problems")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model to use")
    
    args = parser.parse_args()
    
    if args.mode == "compare":
        run_comparison(args.num)
    else:
        # Single mode evaluation
        evaluator = EnhancedEvaluator(model=args.model, mode=args.mode)
        
        # Test with a sample problem
        test_prompt = "Design a buck converter: 24V input to 5V output, 3A load current"
        result = evaluator.evaluate(test_prompt)
        
        print("Response:")
        print(result.get("raw_response", "No response"))
        print("\nExtracted params:")
        print(json.dumps(result.get("params"), indent=2))
