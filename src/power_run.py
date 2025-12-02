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


def validate_circuit(code_file, problem_spec):
    """Validate generated circuit code"""
    validation_result = {
        'syntax_valid': False,
        'simulation_runs': False,
        'output_voltage_ok': False,
        'output_voltage': None,
        'target_voltage': problem_spec['output_voltage'],
        'errors': [],
        'warnings': []
    }
    
    # Read the code
    try:
        with open(code_file, 'r') as f:
            code = f.read()
    except Exception as e:
        validation_result['errors'].append(f"Failed to read file: {e}")
        return validation_result
    
    # Check syntax by compiling
    try:
        compile(code, str(code_file), 'exec')
        validation_result['syntax_valid'] = True
        print("  ✓ Syntax check passed")
    except SyntaxError as e:
        validation_result['errors'].append(f"Syntax error at line {e.lineno}: {e.msg}")
        print(f"  ✗ Syntax error: {e.msg}")
        return validation_result
    
    # Try to execute the circuit (without plotting)
    try:
        # Replace plt.show() to avoid blocking
        test_code = code.replace('plt.show()', 'pass  # Validation mode')
        test_code = test_code.replace('import matplotlib.pyplot as plt', 'import matplotlib\nmatplotlib.use("Agg")\nimport matplotlib.pyplot as plt')
        
        exec_namespace = {}
        exec(test_code, exec_namespace)
        
        validation_result['simulation_runs'] = True
        print("  ✓ Simulation executed successfully")
        
        # Try to extract output voltage for functional validation
        try:
            # Re-run simulation to get analysis result
            import numpy as np
            from PySpice.Spice.Netlist import Circuit
            from PySpice.Unit import u_V, u_A, u_s, u_ms
            
            # Execute code to get circuit and analysis
            exec_namespace_voltage = {}
            test_code_voltage = code.replace('plt.show()', 'pass')
            test_code_voltage = test_code_voltage.replace('import matplotlib.pyplot as plt', 'import matplotlib\nmatplotlib.use("Agg")\nimport matplotlib.pyplot as plt')
            
            # Add code to capture final voltage value
            capture_code = """
import numpy as np
if 'analysis' in dir() and hasattr(analysis, 'Vout'):
    _output_voltage_final = float(analysis['Vout'][-1])
elif 'analysis' in dir() and hasattr(analysis, 'vout'):
    _output_voltage_final = float(analysis['vout'][-1])
else:
    _output_voltage_final = None
"""
            test_code_voltage += "\n" + capture_code
            
            exec(test_code_voltage, exec_namespace_voltage)
            
            if '_output_voltage_final' in exec_namespace_voltage and exec_namespace_voltage['_output_voltage_final'] is not None:
                output_v = exec_namespace_voltage['_output_voltage_final']
                validation_result['output_voltage'] = output_v
                target_v = problem_spec['output_voltage']
                error_pct = abs(output_v - target_v) / target_v * 100
                
                # Accept ±5% tolerance
                if error_pct <= 5.0:
                    validation_result['output_voltage_ok'] = True
                    print(f"  ✓ Output voltage: {output_v:.3f}V (target: {target_v}V, error: {error_pct:.1f}%)")
                else:
                    validation_result['output_voltage_ok'] = False
                    validation_result['errors'].append(f"Output voltage {output_v:.3f}V is {error_pct:.1f}% off target {target_v}V (>5% tolerance)")
                    print(f"  ✗ Output voltage: {output_v:.3f}V (target: {target_v}V, error: {error_pct:.1f}%)")
        except Exception as e:
            # Voltage extraction failed, but simulation ran
            validation_result['warnings'].append(f"Could not extract output voltage: {e}")
            print(f"  ⚠️  Could not validate output voltage: {e}")
        
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        validation_result['errors'].append(error_msg)
        print(f"  ✗ Simulation error: {error_msg}")
        return validation_result
    
    return validation_result


def create_feedback_prompt(original_prompt, code, validation_result, problem_spec):
    """Create feedback prompt for refinement"""
    feedback = f"""Your previous circuit design had the following issues:

## Errors:
"""
    
    for error in validation_result['errors']:
        feedback += f"- {error}\n"
    
    # Add specific guidance for output voltage issues
    if not validation_result.get('output_voltage_ok', True) and validation_result.get('output_voltage') is not None:
        target_v = problem_spec['output_voltage']
        actual_v = validation_result['output_voltage']
        feedback += f"\n## Output Voltage Issue:\n"
        feedback += f"- Target: {target_v}V, Actual: {actual_v:.3f}V\n"
        
        if actual_v > target_v * 1.05:
            feedback += f"- Output is too HIGH. Decrease duty cycle.\n"
            feedback += f"- Check if you applied 1.15× compensation correctly\n"
        elif actual_v < target_v * 0.95:
            feedback += f"- Output is too LOW. Increase duty cycle.\n"
            feedback += f"- Make sure to apply 1.15× compensation: D_actual = (Vout/Vin) × 1.15\n"
            feedback += f"- Verify PWM gate drive is working (not DC)\n"
    
    feedback += f"\n## Previous Code:\n```python\n{code[:500]}...\n```\n\n## Instructions:\nPlease fix the above errors and provide a corrected circuit design.\nRemember:\n- **CRITICAL**: Use PWM gate drive, NOT DC voltage!\n- **CRITICAL**: Apply duty cycle compensation: D_actual = D_ideal × 1.15\n- Use **{{'lambda': value}}** syntax for Python keywords\n- Consider using voltage-controlled switch (VCS) for reliable switching\n- Ensure all components are properly connected\n- Include proper error handling\n\n## Original Task:\n"""
    
    feedback += f"""Design {problem_spec['topology']} converter.
Input: {problem_spec['input_voltage']}V → Output: {problem_spec['output_voltage']}V
Power: {problem_spec['output_power']}W, Frequency: {problem_spec['switching_freq']}kHz

Provide complete, working PySpice code with PWM gate drive and duty cycle compensation.
"""
    
    return feedback


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
    successful_generations = 0
    
    for iteration in range(1, args.num_per_task + 1):
        print(f"\n🔄 Iteration {iteration}/{args.num_per_task}")
        print("-" * 60)
        
        current_prompt = prompt
        retry_count = 0
        
        while retry_count <= args.num_of_retry:
            # Call LLM
            if retry_count == 0:
                print("🤖 Calling LLM...")
            else:
                print(f"🔄 Retry {retry_count}/{args.num_of_retry}...")
                
            llm_response = call_llm(client, current_prompt, model=args.model, temperature=args.temperature)
            
            if not llm_response:
                print("❌ Failed to get response from LLM")
                retry_count += 1
                continue
            
            # Extract code
            print("🔍 Extracting SPICE code...")
            code = extract_spice_code(llm_response)
            
            if not code:
                print("⚠️  No code found in LLM response")
                print("Response preview:", llm_response[:200])
                retry_count += 1
                continue
            
            # Save code
            code_file = save_generated_code(code, output_base, args.task_id, iteration)
            
            # Validate circuit
            print("\n🔬 Validating circuit...")
            validation_result = validate_circuit(code_file, problem)
            
            # Check if validation passed (syntax, simulation, and optionally voltage)
            voltage_check = validation_result.get('output_voltage') is None or validation_result.get('output_voltage_ok', False)
            
            if validation_result['syntax_valid'] and validation_result['simulation_runs'] and voltage_check:
                print(f"\n✅ Generation {iteration} successful!")
                print(f"   Code saved to: {code_file}")
                if validation_result.get('output_voltage') is not None:
                    print(f"   Output voltage: {validation_result['output_voltage']:.3f}V (target: {problem['output_voltage']}V)")
                successful_generations += 1
                break
            else:
                print(f"\n❌ Validation failed (attempt {retry_count + 1}/{args.num_of_retry + 1})")
                
                if retry_count < args.num_of_retry:
                    # Create feedback prompt for refinement
                    print("📝 Creating feedback prompt for refinement...")
                    current_prompt = create_feedback_prompt(prompt, code, validation_result, problem)
                    retry_count += 1
                else:
                    print("⚠️  Max retries reached, moving to next iteration")
                    break
    
    print("\n" + "=" * 60)
    print("✨ Generation complete!")
    print(f"📊 Success rate: {successful_generations}/{args.num_per_task}")
    print(f"📁 Results in: {output_base}")

if __name__ == "__main__":
    main()

