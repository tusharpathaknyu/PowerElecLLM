from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Run the circuit
code = open('gpt_4o/task_1/iteration_1/circuit.py').read()
code = code.replace('plt.show()', 'pass')

namespace = {}
exec(code, namespace)
analysis = namespace['analysis']

# Print all voltages
print("\n" + "="*60)
print("NODE VOLTAGES AT END OF SIMULATION:")
print("="*60)

for node in ['Vin', 'Vsw', 'Vout', 'Vgate']:
    try:
        voltage = float(analysis[node][-1])
        avg_voltage = float(np.mean(analysis[node][-1000:]))
        print(f"{node:10s}: Final={voltage:6.3f}V  Average={avg_voltage:6.3f}V")
    except:
        pass

# Check switching node
vsw = np.array(analysis['Vsw'])
print(f"\nVsw switching:")
print(f"  Min: {np.min(vsw):.3f}V")
print(f"  Max: {np.max(vsw):.3f}V")
print(f"  Range: {np.max(vsw) - np.min(vsw):.3f}V")

# Check gate
vgate = np.array(analysis['Vgate'])
print(f"\nVgate switching:")
print(f"  Min: {np.min(vgate):.3f}V")
print(f"  Max: {np.max(vgate):.3f}V")
print(f"  Is PWM working? {np.max(vgate) - np.min(vgate) > 1}")

print("\n" + "="*60)
print(f"TARGET: 5.000V")
print(f"ACTUAL: {float(analysis['Vout'][-1]):.3f}V")
print(f"ERROR:  {abs(float(analysis['Vout'][-1]) - 5.0):.3f}V")
print("="*60)
