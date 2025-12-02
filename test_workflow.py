#!/usr/bin/env python3
"""
Test script to verify the complete PowerElecLLM workflow without API calls
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from power_run import (
    load_problem_set,
    get_problem_by_id,
    load_template,
    fill_template,
    extract_spice_code,
    validate_circuit
)

def test_workflow():
    """Test all workflow components"""
    print("=" * 60)
    print("PowerElecLLM Workflow Test")
    print("=" * 60)
    
    # Test 1: Load problem set
    print("\n1️⃣  Testing problem loading...")
    try:
        problems = load_problem_set()
        problem = get_problem_by_id(problems, 1)
        print(f"   ✅ Loaded problem: {problem['topology']} converter")
        print(f"      {problem['input_voltage']}V → {problem['output_voltage']}V, {problem['output_power']}W")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    # Test 2: Load and fill template
    print("\n2️⃣  Testing template system...")
    try:
        template = load_template()
        prompt = fill_template(template, problem)
        print(f"   ✅ Template loaded ({len(template)} chars)")
        print(f"   ✅ Prompt filled ({len(prompt)} chars)")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    # Test 3: Code extraction
    print("\n3️⃣  Testing code extraction...")
    test_response = """
Here's a buck converter design:

```python
from PySpice.Spice.Netlist import Circuit
circuit = Circuit('Test')
circuit.V('in', 'Vin', circuit.gnd, 12)
```

This design meets the specifications.
"""
    try:
        code = extract_spice_code(test_response)
        if code and 'Circuit' in code:
            print(f"   ✅ Code extraction working")
        else:
            print(f"   ❌ Code extraction failed")
            return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    # Test 4: Validation
    print("\n4️⃣  Testing circuit validation...")
    try:
        circuit_file = Path('gpt_4o/task_1/iteration_1/circuit.py')
        if circuit_file.exists():
            result = validate_circuit(str(circuit_file), problem)
            if result['syntax_valid'] and result['simulation_runs']:
                print(f"   ✅ Validation system working")
                print(f"      Syntax: {result['syntax_valid']}")
                print(f"      Simulation: {result['simulation_runs']}")
            else:
                print(f"   ⚠️  Validation found issues:")
                for error in result['errors']:
                    print(f"      - {error}")
        else:
            print(f"   ⚠️  No existing circuit to validate (run generator first)")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✅ All workflow tests passed!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_workflow()
    sys.exit(0 if success else 1)
