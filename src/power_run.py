#!/usr/bin/env python3
"""
PowerElecLLM - Main execution script for power electronics circuit generation

This script extends AnalogCoder's approach to power electronics design.
Based on gpt_run.py from AnalogCoder (AAAI'25).
"""

from openai import OpenAI
import argparse
import re
import os
import subprocess
import time
import pandas as pd
import sys
import json
from pathlib import Path

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent

parser = argparse.ArgumentParser(description='Power Electronics LLM Circuit Generator')
parser.add_argument('--model', type=str, default="gpt-4o", help='LLM model to use')
parser.add_argument('--temperature', type=float, default=0.5)
parser.add_argument('--num_per_task', type=int, default=1)
parser.add_argument('--num_of_retry', type=int, default=3)
parser.add_argument('--task_id', type=int, default=1)
parser.add_argument('--api_key', type=str, default=None, help='OpenAI API key (or set OPENAI_API_KEY env var)')

args = parser.parse_args()

# Get API key from args or environment
api_key = args.api_key or os.getenv('OPENAI_API_KEY')
if not api_key:
    print("❌ Error: API key required. Set --api_key or OPENAI_API_KEY environment variable")
    sys.exit(1)

# Initialize OpenAI client
if "gpt" in args.model or "deepseek" in args.model:
    client = OpenAI(api_key=api_key)
    if "deepseek" in args.model:
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
else:
    client = None
    print(f"Model {args.model} not yet supported. Please use GPT models.")


def load_problem_set(problem_file=None):
    """Load problem set from JSON file"""
    if problem_file is None:
        problem_file = PROJECT_ROOT / "benchmarks" / "problem_set.json"
    
    with open(problem_file, 'r') as f:
        problems = json.load(f)
    return problems


def get_problem_by_id(problems, task_id):
    """Get a specific problem by task_id"""
    for problem in problems:
        if problem['task_id'] == task_id:
            return problem
    raise ValueError(f"Problem with task_id={task_id} not found")


def load_template(template_path=None):
    """Load prompt template from file"""
    if template_path is None:
        template_path = PROJECT_ROOT / "templates" / "power_electronics_template.md"
    
    with open(template_path, 'r') as f:
        return f.read()


def fill_template(template, problem_spec):
    """Fill template with problem specifications"""
    filled = template
    
    # Replace placeholders
    replacements = {
        '[TASK]': problem_spec.get('description', f"{problem_spec.get('topology', 'power converter')} converter"),
        '[INPUT_VOLTAGE]': str(problem_spec.get('input_voltage', 'N/A')),
        '[OUTPUT_VOLTAGE]': str(problem_spec.get('output_voltage', 'N/A')),
        '[POWER]': str(problem_spec.get('output_power', 'N/A')),
        '[FREQ]': str(problem_spec.get('switching_freq', 'N/A')),
        '[EFFICIENCY]': str(problem_spec.get('efficiency_target', 'N/A')),
        '[INPUT]': ', '.join(problem_spec.get('input_nodes', ['Vin', 'GND'])),
        '[OUTPUT]': ', '.join(problem_spec.get('output_nodes', ['Vout', 'GND']))
    }
    
    for placeholder, value in replacements.items():
        filled = filled.replace(placeholder, value)
    
    return filled


def call_llm(client, prompt, model="gpt-4o", temperature=0.5):
    """Call LLM API and get response"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a power electronics design expert. Provide complete, working PySpice code."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return None


def extract_spice_code(llm_response):
    """Extract PySpice code from LLM response"""
    if not llm_response:
        return None
    
    # Look for code blocks with ```python or ```spice
    patterns = [
        r'```python\s*(.*?)```',
        r'```spice\s*(.*?)```',
        r'```\s*(.*?)```',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, llm_response, re.DOTALL)
        if matches:
            # Return the first code block found
            code = matches[0].strip()
            # Remove any leading/trailing whitespace
            return code
    
    # If no code block found, try to find code-like content
    # Look for lines that start with common PySpice patterns
    lines = llm_response.split('\n')
    code_lines = []
    in_code = False
    
    for line in lines:
        if any(keyword in line for keyword in ['from PySpice', 'import', 'circuit.', 'Circuit(']):
            in_code = True
        if in_code:
            code_lines.append(line)
            if line.strip() and not line.startswith(' ') and 'def ' in line:
                break
    
    if code_lines:
        return '\n'.join(code_lines)
    
    return None


def save_generated_code(code, output_dir, task_id, iteration):
    """Save generated code to file"""
    output_path = Path(output_dir) / f"task_{task_id}" / f"iteration_{iteration}"
    output_path.mkdir(parents=True, exist_ok=True)
    
    code_file = output_path / "circuit.py"
    with open(code_file, 'w') as f:
        f.write(code)
    
    print(f"✅ Code saved to: {code_file}")
    return code_file


def main():
    print("PowerElecLLM - Power Electronics Circuit Generator")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Task ID: {args.task_id}")
    print("=" * 60)
    
    # Load problem set
    try:
        problems = load_problem_set()
        problem = get_problem_by_id(problems, args.task_id)
        print(f"\n📋 Problem: {problem['topology']} converter")
        print(f"   Input: {problem['input_voltage']}V → Output: {problem['output_voltage']}V")
        print(f"   Power: {problem['output_power']}W, Frequency: {problem['switching_freq']}kHz")
    except Exception as e:
        print(f"❌ Error loading problem: {e}")
        return
    
    # Load template
    try:
        template = load_template()
        prompt = fill_template(template, problem)
        print("\n📝 Template loaded and filled")
    except Exception as e:
        print(f"❌ Error loading template: {e}")
        return
    
    # Generate circuit code for each iteration
    output_base = PROJECT_ROOT / args.model.replace('-', '_')
    
    for iteration in range(1, args.num_per_task + 1):
        print(f"\n🔄 Iteration {iteration}/{args.num_per_task}")
        print("-" * 60)
        
        # Call LLM
        print("🤖 Calling LLM...")
        llm_response = call_llm(client, prompt, model=args.model, temperature=args.temperature)
        
        if not llm_response:
            print("❌ Failed to get response from LLM")
            continue
        
        # Extract code
        print("🔍 Extracting SPICE code...")
        code = extract_spice_code(llm_response)
        
        if not code:
            print("⚠️  No code found in LLM response")
            print("Response preview:", llm_response[:200])
            continue
        
        # Save code
        code_file = save_generated_code(code, output_base, args.task_id, iteration)
        
        print(f"\n✅ Generation {iteration} complete!")
        print(f"   Code saved to: {code_file}")
        
        # TODO: Add PySpice validation here
        # TODO: Add iterative refinement if validation fails
    
    print("\n" + "=" * 60)
    print("✨ Generation complete!")
    print(f"📁 Results in: {output_base}")

if __name__ == "__main__":
    main()

