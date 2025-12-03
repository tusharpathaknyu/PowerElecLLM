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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REFERENCE_TOLERANCE_PCT = 5.0
SANITIZE_IMPORT_PATTERNS = [
    r"^from PySpice\.Spice\.Simulation import .*$",
    r"^import PySpice\.Spice\.Simulation.*$",
    r"^from PySpice\.Probe\.WaveForm import .*$",
]
ESSENTIAL_IMPORTS = [
    "from PySpice.Spice.Netlist import Circuit",
    "from PySpice.Unit import *",
]
UNSUPPORTED_UNIT_PATTERNS = [
    (re.compile(r"@u_degree[s]?"), ""),
]
BLOCKLIST_PATTERNS = [
    (re.compile(r"NotImplementedError"), "code still contains NotImplementedError placeholder"),
    (re.compile(r"TODO"), "code still contains TODO placeholder"),
    (re.compile(r"ExportWaveForm"), "uses unsupported ExportWaveForm helper"),
]

parser = argparse.ArgumentParser(description='Power Electronics LLM Circuit Generator')
parser.add_argument('--model', type=str, default="gpt-4o", help='LLM model to use')
parser.add_argument('--temperature', type=float, default=0.5)
parser.add_argument('--num_per_task', type=int, default=1)
parser.add_argument('--num_of_retry', type=int, default=3)
parser.add_argument('--task_id', type=int, default=1)
parser.add_argument('--api_key', type=str, default=None, help='OpenAI API key (or set OPENAI_API_KEY env var)')
parser.add_argument('--run_reference_tests', action='store_true', help='Run built-in buck/boost regression suite before generation')

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


def sanitize_generated_code(code: str):
    """Remove unsupported imports and ensure essential ones exist."""
    warnings = []
    sanitized_lines = []
    for line in code.splitlines():
        stripped = line.strip()
        if any(re.match(pattern, stripped) for pattern in SANITIZE_IMPORT_PATTERNS):
            warnings.append(f"Removed unsupported import: {stripped}")
            continue
        sanitized_lines.append(line)

    sanitized_code = "\n".join(sanitized_lines)

    for pattern, replacement in UNSUPPORTED_UNIT_PATTERNS:
        sanitized_code_new, count = pattern.subn(replacement, sanitized_code)
        if count:
            warnings.append(f"Replaced unsupported unit pattern '{pattern.pattern}' {count} time(s)")
            sanitized_code = sanitized_code_new

    for imp in ESSENTIAL_IMPORTS:
        if imp not in sanitized_code:
            sanitized_code = f"{imp}\n" + sanitized_code
            warnings.append(f"Inserted missing import: {imp}")

    return sanitized_code.strip() + "\n", warnings


def autofix_diode_models(code: str):
    warnings = []
    lines = code.splitlines()
    diode_lines = []
    diode_call_pattern = re.compile(r"circuit\s*\.\s*d\s*\(", re.IGNORECASE)
    diode_models_in_code = set(
        re.findall(r"circuit\.model\(\s*['\"]([^'\"]+)['\"]\s*,\s*['\"]d['\"]", code, re.IGNORECASE)
    )

    for idx, line in enumerate(lines):
        if diode_call_pattern.search(line):
            diode_lines.append(idx)

    if not diode_lines:
        return code, warnings

    default_model_name = 'DMOD_AUTO'
    diode_model_names = set()

    for idx in diode_lines:
        line = lines[idx]
        model_match = re.search(r"model\s*=\s*['\"]([^'\"]+)['\"]", line)
        if model_match:
            model_name = model_match.group(1)
            diode_model_names.add(model_name)
            continue

        # No model specified; append default model argument
        stripped = line.rstrip()
        if stripped.endswith(')'):
            stripped = stripped[:-1] + f", model='{default_model_name}')"
        else:
            stripped = stripped + f", model='{default_model_name}'"
        lines[idx] = stripped
        warnings.append("Inserted default diode model reference 'DMOD_AUTO'")
        diode_model_names.add(default_model_name)

    missing_definitions = [
        name for name in diode_model_names if name and name not in diode_models_in_code
    ]

    if missing_definitions:
        insert_idx = None
        for idx, line in enumerate(lines):
            if 'Circuit(' in line:
                insert_idx = idx + 1
                break
        if insert_idx is None:
            insert_idx = len(lines)

        for name in missing_definitions:
            snippet = (
                f"circuit.model('{name}', 'D', **{{'is': 1e-9}}, Rs=0.05, N=1.5)"
            )
            lines.insert(insert_idx, snippet)
            insert_idx += 1
            warnings.append(f"Inserted diode model definition for '{name}'")

    return "\n".join(lines) + "\n", warnings


def detect_blocklist_issues(code: str):
    issues = []
    for pattern, message in BLOCKLIST_PATTERNS:
        if pattern.search(code):
            issues.append(message)
    return issues


def autofix_missing_diode(code: str, topology: str):
    """Auto-insert freewheeling diode for buck converter if missing."""
    warnings = []
    topology_lower = topology.lower()
    
    # Only applies to buck converters
    if 'buck' not in topology_lower:
        return code, warnings
    
    # Check if diode already present
    diode_call_pattern = re.compile(r"circuit\s*\.\s*d\s*\(", re.IGNORECASE)
    if diode_call_pattern.search(code):
        return code, warnings
    
    # Look for switch node (Vsw) or similar patterns to identify insertion point
    lines = code.splitlines()
    insert_idx = None
    vsw_pattern = re.compile(r"['\"]Vsw['\"]|['\"]vsw['\"]|['\"]VSW['\"]|['\"]SW['\"]|['\"]sw['\"]")
    
    # Find line with inductor (often connected to Vsw)
    inductor_pattern = re.compile(r"circuit\s*\.\s*l\s*\(", re.IGNORECASE)
    
    for idx, line in enumerate(lines):
        if inductor_pattern.search(line) and vsw_pattern.search(line):
            insert_idx = idx
            break
    
    # Fallback: insert after any VCS (voltage-controlled switch) definition
    if insert_idx is None:
        vcs_pattern = re.compile(r"circuit\s*\.\s*vcs\s*\(", re.IGNORECASE)
        for idx, line in enumerate(lines):
            if vcs_pattern.search(line):
                insert_idx = idx + 1
                break
    
    # Final fallback: insert before the load resistor
    if insert_idx is None:
        load_pattern = re.compile(r"circuit\s*\.\s*r\s*\(.*['\"]Rload['\"]", re.IGNORECASE)
        for idx, line in enumerate(lines):
            if load_pattern.search(line):
                insert_idx = idx
                break
    
    # Last resort: insert after 'Circuit(' line
    if insert_idx is None:
        for idx, line in enumerate(lines):
            if 'Circuit(' in line:
                insert_idx = idx + 1
                break
    
    if insert_idx is None:
        warnings.append("Could not find insertion point for freewheeling diode")
        return code, warnings
    
    # Insert canonical diode snippet
    diode_model_snippet = "circuit.model('DMOD', 'D', **{'is': 1e-9}, Rs=0.05, N=1.5)"
    diode_element_snippet = "circuit.D('D1', circuit.gnd, 'Vsw', model='DMOD')  # Freewheeling diode"
    
    # Check if model already defined
    model_exists = bool(re.search(r"circuit\.model\(['\"]DMOD['\"]", code, re.IGNORECASE))
    
    # Determine proper indentation by looking at circuit statements after insert point
    indent = ""  # Default to no indent (module level)
    for line in lines[insert_idx:min(insert_idx+10, len(lines))]:
        stripped = line.lstrip()
        if stripped.startswith('circuit.'):
            indent = line[:len(line) - len(stripped)]
            break
    # Also check lines before insert point
    if indent == "":
        for line in lines[max(0, insert_idx-5):insert_idx]:
            stripped = line.lstrip()
            if stripped.startswith('circuit.'):
                indent = line[:len(line) - len(stripped)]
                break
    
    insertions = []
    if not model_exists:
        insertions.append(indent + diode_model_snippet)
    insertions.append(indent + diode_element_snippet)
    
    for i, snippet in enumerate(insertions):
        lines.insert(insert_idx + i, snippet)
    
    warnings.append("Auto-inserted freewheeling diode (DMOD + D1) for buck topology")
    
    return "\n".join(lines) + "\n", warnings


def autofix_buck_diode_polarity(code: str, topology: str):
    """Fix buck diode polarity: should be (GND, Vsw), not (Vout, Vsw)."""
    warnings = []
    topology_lower = topology.lower()
    
    # Only applies to buck converters
    if 'buck' not in topology_lower:
        return code, warnings
    
    # Find diode lines with wrong polarity: circuit.D(..., 'Vout', 'Vsw', ...) or similar
    # Correct buck freewheel: circuit.D(..., circuit.gnd, 'Vsw', ...)
    # Wrong pattern: anode at Vout, cathode at Vsw
    wrong_patterns = [
        # circuit.D(name, 'Vout', 'Vsw', ...) - wrong
        (re.compile(r"(circuit\s*\.\s*[dD]\s*\([^)]*['\"]Vout['\"])\s*,\s*(['\"]Vsw['\"])", re.IGNORECASE),
         r"\1_WRONG, circuit.gnd, \2"),
    ]
    
    lines = code.splitlines()
    modified = False
    
    for idx, line in enumerate(lines):
        # Check for common wrong pattern: D(..., 'Vout', 'Vsw', ...)
        # The anode should be GND for buck freewheel, not Vout
        if re.search(r"circuit\s*\.\s*[dD]\s*\(", line, re.IGNORECASE):
            # Extract the diode call arguments
            match = re.search(r"circuit\s*\.\s*[dD]\s*\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,\)]+)", line, re.IGNORECASE)
            if match:
                name_arg = match.group(1).strip()
                anode_arg = match.group(2).strip()
                cathode_arg = match.group(3).strip()
                
                # Check if anode is Vout (wrong for buck) and cathode is Vsw
                anode_is_vout = "'vout'" in anode_arg.lower() or '"vout"' in anode_arg.lower()
                cathode_is_vsw = "'vsw'" in cathode_arg.lower() or '"vsw"' in cathode_arg.lower()
                
                if anode_is_vout and cathode_is_vsw:
                    # Fix: change anode from Vout to circuit.gnd
                    old_call = match.group(0)
                    new_call = f"circuit.D({name_arg}, circuit.gnd, {cathode_arg}"
                    lines[idx] = line.replace(old_call, new_call)
                    warnings.append(f"Fixed buck diode polarity: changed anode from Vout to GND")
                    modified = True
    
    if modified:
        return "\n".join(lines) + "\n", warnings
    
    return code, warnings


def autofix_boost_switch_placement(code: str, topology: str):
    """Fix boost converter switch: should be Vsw→GND, not Vsw→Vout."""
    warnings = []
    topology_lower = topology.lower()
    
    # Only applies to boost converters
    if 'boost' not in topology_lower:
        return code, warnings
    
    lines = code.splitlines()
    modified = False
    
    # Pattern: VCS(..., 'Vsw', 'Vout', ...) should be VCS(..., 'Vsw', circuit.gnd, ...)
    vcs_pattern = re.compile(r"circuit\s*\.\s*vcs\s*\(", re.IGNORECASE)
    
    for idx, line in enumerate(lines):
        if vcs_pattern.search(line):
            # Check if second terminal is Vout (wrong) instead of GND
            # VCS('name', terminal1, terminal2, ctrl+, ctrl-, ...)
            # For boost low-side switch: VCS('S1', 'Vsw', circuit.gnd, 'Vgate', circuit.gnd)
            if "'vout'" in line.lower() or '"vout"' in line.lower():
                # Replace Vout with circuit.gnd for the switch terminal
                # This is a heuristic - look for 'Vsw', 'Vout' pattern
                new_line = re.sub(
                    r"(['\"]Vsw['\"])\s*,\s*['\"]Vout['\"]",
                    r"\1, circuit.gnd",
                    line,
                    flags=re.IGNORECASE
                )
                if new_line != line:
                    lines[idx] = new_line
                    warnings.append("Fixed boost switch: terminal changed from Vout to GND (low-side)")
                    modified = True
    
    if modified:
        return "\n".join(lines) + "\n", warnings
    
    return code, warnings


def autofix_boost_capacitor_ic(code: str, topology: str):
    """Ensure boost converter output capacitor starts at 0V."""
    warnings = []
    topology_lower = topology.lower()
    
    # Only applies to boost converters
    if 'boost' not in topology_lower:
        return code, warnings
    
    lines = code.splitlines()
    modified = False
    
    # Find capacitor connected to Vout
    cap_pattern = re.compile(r"circuit\s*\.\s*c\s*\(", re.IGNORECASE)
    
    for idx, line in enumerate(lines):
        if cap_pattern.search(line) and ('vout' in line.lower()):
            # Check if initial_condition is already set
            if 'initial_condition' not in line.lower():
                # Add initial_condition=0@u_V
                stripped = line.rstrip()
                if stripped.endswith(')'):
                    stripped = stripped[:-1] + ", initial_condition=0@u_V)"
                    lines[idx] = stripped
                    warnings.append("Added initial_condition=0V to boost output capacitor")
                    modified = True
    
    if modified:
        return "\n".join(lines) + "\n", warnings
    
    return code, warnings


def detect_structural_issues(code: str, problem_spec: dict):
    issues = []
    code_lower = code.lower()
    topology = problem_spec.get('topology', '').lower()

    if 'pulsevoltagesource' not in code_lower:
        issues.append("Gate drive must use PulseVoltageSource for PWM")

    if 'sinusoidalvoltagesource' in code_lower:
        issues.append("Gate drive uses SinusoidalVoltageSource; switch to PulseVoltageSource")

    diode_call_pattern = re.compile(r"circuit\s*\.\s*d\s*\(", re.IGNORECASE)
    diode_present = bool(diode_call_pattern.search(code))
    if 'buck' in topology and not diode_present:
        issues.append("Buck converter missing freewheeling diode")
    if diode_present:
        has_diode_model = bool(
            re.search(r"circuit\.model\(.*['\"]d['\"]", code, re.IGNORECASE)
        )
        if not has_diode_model:
            issues.append("Diode is instantiated but no diode model (circuit.model(... 'D' ...)) is defined")

    for line in code.splitlines():
        if '@ u_Ohm' in line and any(op in line for op in ['**', '/']):
            issues.append("Compute load resistance as a float before applying units (no expressions inside @u_Ω)")
            break

    return issues


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

        # Try to extract output voltage from the executed namespace
        analysis_obj = exec_namespace.get('analysis')
        if analysis_obj is None:
            validation_result['warnings'].append("Analysis object not found; cannot measure Vout")
        else:
            try:
                import numpy as np

                vout_trace = None
                # Extended node candidates including negative output nodes
                node_candidates = ['Vout', 'vout', 'VOUT', 'out', 'Vo', 'Vout_neg', 'vout_neg', 'V_out', 'output']
                for node in node_candidates:
                    try:
                        vout_trace = np.array(analysis_obj[node])
                        if vout_trace.size:
                            break
                    except Exception:
                        vout_trace = None

                if vout_trace is None or vout_trace.size == 0:
                    raise ValueError("Vout waveform not found in analysis")

                window = max(int(0.1 * len(vout_trace)), 10)
                output_v = float(np.mean(vout_trace[-window:]))
                validation_result['output_voltage'] = output_v
                target_v = problem_spec['output_voltage']
                
                # Handle negative target voltages properly
                if target_v != 0:
                    error_pct = abs(output_v - target_v) / abs(target_v) * 100
                else:
                    error_pct = abs(output_v) * 100  # If target is 0, any output is error

                if error_pct <= 5.0:
                    validation_result['output_voltage_ok'] = True
                    print(f"  ✓ Output voltage: {output_v:.3f}V (target: {target_v}V, error: {error_pct:.1f}%)")
                else:
                    validation_result['output_voltage_ok'] = False
                    validation_result['errors'].append(
                        f"Output voltage {output_v:.3f}V is {error_pct:.1f}% off target {target_v}V (>5% tolerance)"
                    )
                    print(f"  ✗ Output voltage: {output_v:.3f}V (target: {target_v}V, error: {error_pct:.1f}%)")
            except Exception as e:
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


def run_reference_guard(tolerance_pct: float = REFERENCE_TOLERANCE_PCT) -> bool:
    """Run deterministic buck/boost regression suite to ensure template stability."""
    print("\n🧪 Running reference regression suite...")
    try:
        from reference_tests.run_reference_tests import (
            run_reference_tests,
            print_summary,
            all_within_tolerance,
        )
    except Exception as exc:  # pragma: no cover - defensive guard
        print(f"⚠️  Could not import reference tests: {exc}")
        return False

    try:
        results = run_reference_tests()
        print_summary(results, tolerance_pct)
    except Exception as exc:
        print(f"❌ Reference regression execution failed: {exc}")
        return False

    if not all_within_tolerance(results, tolerance_pct):
        print("❌ Reference regression failed tolerance check. Investigate before generating new circuits.")
        return False

    print("✅ Reference regression passed; proceeding with generation.")
    return True


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

    if args.run_reference_tests:
        if not run_reference_guard():
            print("🚫 Aborting generation because reference suite failed.")
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

            code, sanitize_warnings = sanitize_generated_code(code)
            for warn in sanitize_warnings:
                print(f"  ⚠️  {warn}")

            code, diode_warnings = autofix_diode_models(code)
            for warn in diode_warnings:
                print(f"  ⚙️  {warn}")

            # Auto-fix missing freewheeling diode for buck converters
            topology = problem.get('topology', '')
            code, diode_insert_warnings = autofix_missing_diode(code, topology)
            for warn in diode_insert_warnings:
                print(f"  ⚙️  {warn}")

            # Auto-fix buck diode polarity (anode should be GND, not Vout)
            code, polarity_warnings = autofix_buck_diode_polarity(code, topology)
            for warn in polarity_warnings:
                print(f"  ⚙️  {warn}")

            # Auto-fix boost switch placement (should be Vsw→GND, not Vsw→Vout)
            code, boost_switch_warnings = autofix_boost_switch_placement(code, topology)
            for warn in boost_switch_warnings:
                print(f"  ⚙️  {warn}")

            # Auto-fix boost capacitor initial condition (should start at 0V)
            code, boost_cap_warnings = autofix_boost_capacitor_ic(code, topology)
            for warn in boost_cap_warnings:
                print(f"  ⚙️  {warn}")

            blocklist_issues = detect_blocklist_issues(code)
            if blocklist_issues:
                print("  ❌ Code rejected due to placeholders:")
                for issue in blocklist_issues:
                    print(f"     - {issue}")
                reminder = "\n\nReminder: Provide complete, working PySpice code with no TODOs or NotImplementedError placeholders."
                current_prompt = current_prompt + reminder
                retry_count += 1
                continue

            structural_issues = detect_structural_issues(code, problem)
            if structural_issues:
                print("  ❌ Code rejected due to structural issues:")
                for issue in structural_issues:
                    print(f"     - {issue}")
                reminder = "\n\nReminder: Buck designs must include a freewheeling diode and use PulseVoltageSource PWM. Always compute numeric load values before applying units."
                current_prompt = current_prompt + reminder
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

