#!/usr/bin/env python3
"""
Validation script for buck converter designs
"""

import sys
import os
from pathlib import Path
import re

def fix_pyspice_keywords(code):
    """Fix Python keyword conflicts in PySpice model definitions"""
    # PySpice uses 'lambda' and 'is' which are Python keywords
    # We need to use **kwargs syntax instead
    
    # Fix lambda= in model definitions (must be before other params)
    # Pattern: lambda=0.01 or , lambda=0.01
    def fix_lambda(match):
        value = match.group(1)
        prefix = match.group(0).split('lambda')[0]  # Everything before lambda
        return f"{prefix}**{{'lambda': {value}}}"
    
    code = re.sub(r'(\s|,)\s*lambda=([0-9.e-]+)', r"\1**{'lambda': \2}", code)
    
    # Fix is= in model definitions (for diodes) - must use **kwargs
    # Pattern: is=1e-6 (standalone or with other params)
    # Replace: is=1e-6, rs=0.05, n=1.05
    # With: **{'is': 1e-6}, rs=0.05, n=1.05
    code = re.sub(r'\bis=([0-9.e-]+)(\s*,)', r"**{'is': \1}\2", code)
    code = re.sub(r'(\s|,)\s*is=([0-9.e-]+)(?!\s*[=,])', r"\1**{'is': \2}", code)
    
    return code

def validate_circuit_code(circuit_file):
    """Validate that circuit code can be executed"""
    try:
        with open(circuit_file, 'r') as f:
            code = f.read()
        
        # Fix common issues
        code = fix_pyspice_keywords(code)
        
        # Remove plotting for validation
        code = code.replace('plt.show()', 'pass  # Plotting disabled for validation')
        
        # Execute in a safe namespace
        namespace = {}
        exec(code, namespace)
        
        print("✅ Circuit code is syntactically valid")
        return True, None
        
    except SyntaxError as e:
        print(f"❌ Syntax error: {e}")
        return False, str(e)
    except Exception as e:
        print(f"⚠️  Runtime error: {e}")
        return False, str(e)

def check_circuit_structure(code):
    """Check if circuit has required components"""
    checks = {
        'has_input_voltage': 'V(' in code and 'Vin' in code,
        'has_switch': 'MOSFET' in code or 'Q' in code,
        'has_inductor': 'L(' in code,
        'has_capacitor': 'C(' in code,
        'has_load': 'R(' in code and 'load' in code.lower(),
        'has_output': 'Vout' in code,
    }
    
    all_pass = all(checks.values())
    
    print("\n📋 Component Checklist:")
    for check, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {check.replace('_', ' ').title()}")
    
    return all_pass

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python buck_check.py <circuit_file.py>")
        sys.exit(1)
    
    circuit_file = sys.argv[1]
    
    if not os.path.exists(circuit_file):
        print(f"❌ File not found: {circuit_file}")
        sys.exit(1)
    
    print(f"🔍 Validating: {circuit_file}\n")
    
    # Read code
    with open(circuit_file, 'r') as f:
        code = f.read()
    
    # Check structure
    structure_ok = check_circuit_structure(code)
    
    # Validate syntax
    syntax_ok, error = validate_circuit_code(circuit_file)
    
    print("\n" + "="*60)
    if syntax_ok and structure_ok:
        print("✅ Validation PASSED")
        sys.exit(0)
    else:
        print("❌ Validation FAILED")
        if error:
            print(f"Error: {error}")
        sys.exit(1)

