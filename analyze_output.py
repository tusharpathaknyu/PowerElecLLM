#!/usr/bin/env python3
"""
Quick analysis of generated circuit output
"""
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

print("=" * 60)
print("Circuit Analysis Tool")
print("=" * 60)

# Load and run the circuit
print("\n📊 Running simulation...")
code = open('gpt_4o/task_1/iteration_1/circuit.py').read()
code = code.replace('plt.show()', 'pass')
code = code.replace('import matplotlib.pyplot as plt', 'import matplotlib\nmatplotlib.use("Agg")\nimport matplotlib.pyplot as plt')

namespace = {'__name__': '__main__'}
try:
    exec(code, namespace)
    analysis = namespace.get('analysis')
    
    # Debug: check what variables we got
    if not analysis:
        print("⚠️  'analysis' not found, checking namespace...")
        for key in namespace:
            if 'analysis' in key.lower() or 'simulator' in key.lower():
                print(f"   Found: {key}")
                analysis = namespace[key]
                break
    
    if analysis:
        # Extract data
        time = np.array(analysis['Vout'].abscissa)
        vout = np.array(analysis['Vout'])
        
        # Calculate metrics
        final_vout = float(vout[-1])
        avg_vout = float(np.mean(vout[-1000:]))  # Last 1ms
        max_vout = float(np.max(vout))
        min_vout = float(np.min(vout[-1000:]))
        ripple = float(max_vout - min_vout)
        error = abs(avg_vout - 5.0)
        error_pct = (error / 5.0) * 100
        
        # Display results
        print("\n✅ Simulation completed successfully!")
        print("\n" + "=" * 60)
        print("RESULTS:")
        print("=" * 60)
        print(f"🎯 Target Output:           5.000 V")
        print(f"📈 Actual Output (final):   {final_vout:.3f} V")
        print(f"📊 Actual Output (avg):     {avg_vout:.3f} V")
        print(f"📉 Voltage Ripple:          {ripple*1000:.1f} mV")
        print(f"⚠️  Error from target:      {error*1000:.1f} mV ({error_pct:.2f}%)")
        
        # Pass/Fail
        print("\n" + "=" * 60)
        print("VALIDATION:")
        print("=" * 60)
        
        if abs(avg_vout - 5.0) < 0.25:  # 5% tolerance
            print("✅ PASS: Output voltage within ±5% tolerance")
        else:
            print("❌ FAIL: Output voltage outside tolerance")
            
        if ripple < 0.05:  # 50mV ripple
            print("✅ PASS: Ripple within acceptable range (<50mV)")
        else:
            print("⚠️  WARN: Ripple higher than typical (<50mV)")
        
        # Save plot
        plt.figure(figsize=(12, 6))
        plt.plot(time * 1000, vout, 'b-', linewidth=2, label='Vout')
        plt.axhline(y=5.0, color='r', linestyle='--', linewidth=2, label='Target (5V)')
        plt.xlabel('Time (ms)', fontsize=12)
        plt.ylabel('Output Voltage (V)', fontsize=12)
        plt.title('Buck Converter Output - 12V → 5V, 10W', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        
        # Add text box with stats
        textstr = f'Vout: {avg_vout:.3f}V\nRipple: {ripple*1000:.1f}mV\nError: {error_pct:.2f}%'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes,
                fontsize=11, verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        plt.savefig('circuit_analysis.png', dpi=150, bbox_inches='tight')
        
        print("\n" + "=" * 60)
        print(f"📊 Plot saved to: circuit_analysis.png")
        print("=" * 60)
        
    else:
        print("❌ Could not extract analysis data")
        
except Exception as e:
    print(f"❌ Error running simulation: {e}")
    import traceback
    traceback.print_exc()
