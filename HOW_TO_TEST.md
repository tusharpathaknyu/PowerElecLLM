# 🧪 How to Test Your Generated Circuits

## 🎯 Quick Start

### **Option 1: Run the Circuit Directly**
```bash
# Run the generated circuit
python gpt_4o/task_1/iteration_1/circuit.py
```

**What happens:**
- ✅ Circuit simulates for 5ms
- ✅ Matplotlib window opens showing output voltage
- ✅ You see the buck converter voltage waveform

**Expected output:**
- A plot window showing voltage rising from 0V to ~5V
- Title: "Buck Converter Output Voltage"

---

### **Option 2: Run with Analysis**
```bash
# Run with detailed output
python -u gpt_4o/task_1/iteration_1/circuit.py
```

---

### **Option 3: Save Plot Instead of Displaying**

Create a modified test script:

```bash
# Create test script
cat > test_circuit.py << 'EOF'
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Load and run the circuit
exec(open('gpt_4o/task_1/iteration_1/circuit.py').read().replace('plt.show()', 'plt.savefig("output.png"); print("✅ Plot saved to output.png")'))
EOF

python test_circuit.py
```

Then view the image:
```bash
open output.png  # macOS
```

---

## 📊 What to Check in the Output

### **1. Voltage Level**
- **Target:** 5V output (for task_1)
- **Check:** Final voltage in plot should be ~5V
- **Tolerance:** ±5% (4.75V - 5.25V is good)

### **2. Rise Time**
- **Expected:** Voltage should rise smoothly
- **Check:** No oscillations or spikes
- **Good sign:** Smooth exponential rise

### **3. Steady State**
- **Expected:** Voltage stabilizes
- **Check:** Flat line at end of simulation
- **Good sign:** Minimal ripple (<1%)

### **4. Component Values**
Look at the code to verify reasonable values:
```python
circuit.L('L1', 'Vsw', 'Vout', 24.3@u_uH)  # Inductor: ~20-30µH is typical
circuit.C('C1', 'Vout', circuit.gnd, 100@u_uF)  # Cap: 100µF is good
circuit.R('load', 'Vout', circuit.gnd, 2.5@u_Ω)  # Load: 2.5Ω for 10W at 5V ✓
```

---

## 🔍 Advanced Testing

### **Extract Numerical Values**

```python
# Create analysis script
cat > analyze_circuit.py << 'EOF'
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *
import numpy as np

# Run circuit (suppress plotting)
code = open('gpt_4o/task_1/iteration_1/circuit.py').read()
code = code.replace('plt.show()', 'pass')
code = code.replace('import matplotlib.pyplot as plt', 'import matplotlib\nmatplotlib.use("Agg")\nimport matplotlib.pyplot as plt')

namespace = {}
exec(code, namespace)
analysis = namespace.get('analysis')

if analysis:
    # Extract final voltage
    vout = float(analysis['Vout'][-1])
    vout_avg = float(np.mean(analysis['Vout'][-1000:]))  # Last 1ms average
    
    print(f"📊 Circuit Analysis Results:")
    print(f"   Final Vout: {vout:.3f}V")
    print(f"   Average Vout (last 1ms): {vout_avg:.3f}V")
    print(f"   Target: 5.000V")
    print(f"   Error: {abs(vout_avg - 5.0):.3f}V ({abs(vout_avg - 5.0)/5.0*100:.1f}%)")
    
    if abs(vout_avg - 5.0) < 0.25:
        print("   ✅ PASS: Within 5% tolerance")
    else:
        print("   ❌ FAIL: Outside tolerance")
EOF

python analyze_circuit.py
```

---

## 🎨 Better Visualization

### **Enhanced Plot Script**

```python
cat > plot_detailed.py << 'EOF'
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *
import matplotlib.pyplot as plt
import numpy as np

# Run simulation
code = open('gpt_4o/task_1/iteration_1/circuit.py').read()
code = code.replace('plt.show()', 'pass')
code = code.replace('plt.figure(figsize=(10, 5))', 'pass')

namespace = {}
exec(code, namespace)
analysis = namespace['analysis']

# Create detailed plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Plot 1: Output voltage
time = np.array(analysis['Vout'].abscissa)
vout = np.array(analysis['Vout'])

ax1.plot(time * 1000, vout, 'b-', linewidth=2)
ax1.axhline(y=5.0, color='r', linestyle='--', label='Target (5V)')
ax1.set_xlabel('Time (ms)')
ax1.set_ylabel('Output Voltage (V)')
ax1.set_title('Buck Converter - Output Voltage')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Plot 2: Zoom on steady state
steady_start = int(len(time) * 0.8)  # Last 20%
ax2.plot(time[steady_start:] * 1000, vout[steady_start:], 'g-', linewidth=2)
ax2.axhline(y=5.0, color='r', linestyle='--', label='Target')
ax2.set_xlabel('Time (ms)')
ax2.set_ylabel('Output Voltage (V)')
ax2.set_title('Steady State Detail (Last 20%)')
ax2.grid(True, alpha=0.3)
ax2.legend()

# Add statistics
avg_vout = np.mean(vout[steady_start:])
ripple = np.max(vout[steady_start:]) - np.min(vout[steady_start:])
error = abs(avg_vout - 5.0)

stats_text = f'Avg: {avg_vout:.3f}V\nRipple: {ripple*1000:.1f}mV\nError: {error*1000:.1f}mV'
ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('detailed_analysis.png', dpi=150)
print('✅ Detailed plot saved to: detailed_analysis.png')
plt.show()
EOF

python plot_detailed.py
```

---

## 🎯 Quick Test Checklist

Run through these checks:

```bash
# 1. Does it run without errors?
python gpt_4o/task_1/iteration_1/circuit.py
# ✅ Should see a plot window

# 2. Check the code quality
python problem_check/buck_check.py gpt_4o/task_1/iteration_1/circuit.py
# ✅ Should pass all checks

# 3. Run the workflow test
python test_workflow.py
# ✅ All tests should pass
```

---

## 🐛 Troubleshooting

### **Plot Window Doesn't Appear**
```bash
# Use non-interactive backend
export MPLBACKEND=Agg
python gpt_4o/task_1/iteration_1/circuit.py
# Check for output.png file
```

### **Ngspice Warnings**
```
Warning: can't find the initialization file spinit.
Unsupported Ngspice version 45
```
**These are non-critical** - the simulation still works!

### **Import Errors**
```bash
# Reinstall dependencies
pip install pyspice matplotlib numpy
```

---

## 📈 Success Criteria

Your circuit is working well if:

- ✅ **Voltage:** Output is 4.75V - 5.25V (within 5%)
- ✅ **Stability:** Voltage settles and stays constant
- ✅ **Ripple:** Less than 50mV peak-to-peak
- ✅ **Components:** Values are reasonable (L: 10-50µH, C: 50-200µF)
- ✅ **No errors:** Code runs without Python exceptions

---

## 🚀 Test Multiple Iterations

If you generated multiple circuits:

```bash
# Test all iterations
for i in {1..3}; do
    echo "Testing iteration $i..."
    python gpt_4o/task_1/iteration_$i/circuit.py
    sleep 2
done
```

---

## 💡 Pro Tips

1. **Close plot windows** between runs (or they stack up)
2. **Use non-interactive backend** for batch testing
3. **Save plots** instead of displaying for automated testing
4. **Check component values** make physical sense
5. **Compare iterations** to see design variations

---

## 📝 What Good Output Looks Like

```
Time: 0ms     Vout: 0.0V    (starting)
Time: 1ms     Vout: 3.2V    (rising)
Time: 2ms     Vout: 4.7V    (approaching target)
Time: 3ms     Vout: 4.95V   (settling)
Time: 4ms     Vout: 5.02V   (steady state)
Time: 5ms     Vout: 5.01V   (stable) ✅
```

**Ripple:** <1% (less than 50mV)
**Error:** <5% (within 250mV of 5V target)

---

**Ready to test? Just run:**
```bash
python gpt_4o/task_1/iteration_1/circuit.py
```

A plot window will pop up showing your buck converter output! 📊
