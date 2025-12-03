You aim to design a power electronic circuit topology for a given specification.
Please ensure your designed circuit topology works properly and achieves the design requirements.

## Power Electronics Design Principles

### 1. Topology Selection
Choose the appropriate converter topology based on requirements:
- **Buck Converter**: Step-down (Vin > Vout), non-isolated
  - Switch: High-side (between Vin and switching node)
  - Diode: Low-side (from GND to switching node)
  
- **Boost Converter**: Step-up (Vin < Vout), non-isolated
  - **CRITICAL:** Switch: Low-side (between switching node and GND)
  - **CRITICAL:** Diode direction: Anode at switching node, Cathode at output
  - **CRITICAL:** Output cap MUST start at 0V (not pre-charged)
  - Use larger inductors (2-5× larger than buck for same specs)
  
- **Buck-Boost**: Bidirectional, non-isolated
- **Flyback**: Isolated, low-medium power (<100W)
- **Forward**: Isolated, medium power
- **LLC Resonant**: High efficiency, high power

### 2. Component Sizing Formulas

**Inductor (L)**:
- Buck: L = (Vin - Vout) × D / (ΔI_L × f_sw)
  - Typical values: 10-50µH for 100-500kHz
- Boost: L = Vin × D / (ΔI_L × f_sw)  
  - **Use 2-5× larger than buck**: 50-200µH for stability
  - Larger inductor = better energy storage, smoother startup
- Where: D = duty cycle, ΔI_L = inductor ripple current (20-40% of I_out), f_sw = switching frequency

**Output Capacitor (C)**:
- C = I_out × D / (ΔV_out × f_sw)
- Where: ΔV_out = output voltage ripple (typically <1% of Vout)

**Duty Cycle Compensation**:
- **CRITICAL**: Real circuits have losses that reduce output voltage
- **You MUST calculate actual losses**, not use a fixed multiplier

**Loss Components to Consider**:
1. **Diode forward voltage drop**:
   - Schottky diode: 0.3-0.5V (typical)
   - Standard diode: 0.7-1.0V
   - SiC diode: 1.2-1.5V

2. **Conduction losses**:
   - Switch resistance: V_switch = R_ds_on × I_load
   - Inductor DCR: V_inductor = R_DCR × I_load
   - PCB traces: ~0.01-0.05Ω (can usually ignore for initial design)

3. **Calculate compensated duty cycle**:
   ```
   # For Buck converter:
   I_load = P_out / V_out
   V_diode = 0.5  # Schottky typical
   V_switch = R_ds_on × I_load  # Use switch model's R_ds_on
   V_inductor = R_DCR × I_load  # Use inductor's DCR
   
   V_loss_total = V_diode + V_switch + V_inductor
   D_ideal = V_out / V_in
   D_compensated = (V_out + V_loss_total) / V_in
   ```

**Example Calculation** (12V→5V, 10W):
```python
I_load = 10 / 5  # 2A
V_diode = 0.5  # Schottky
R_ds_on = 0.01  # 10mΩ switch
R_DCR = 0.02    # 20mΩ inductor

V_switch = 0.01 * 2 = 0.02V
V_inductor = 0.02 * 2 = 0.04V
V_loss_total = 0.5 + 0.02 + 0.04 = 0.56V

D_ideal = 5 / 12 = 0.417 (41.7%)
D_compensated = (5 + 0.56) / 12 = 0.463 (46.3%)
```

**Switching Frequency (f_sw)**:
- GaN devices: 100kHz - 1MHz (typically 200-500kHz)
- Higher frequency = smaller components but more losses

### 3. GaN Device Characteristics
- Low gate charge (Qg) enables fast switching
- Low on-resistance (Rds_on) reduces conduction losses
- Gate drive voltage: 5-6V (not 12V like Si MOSFETs)
- Fast switching reduces switching losses

### 4. Safety & Protection
- Over-voltage protection (OVP)
- Over-current protection (OCP)  
- Soft-start circuit
- Proper gate drive with dead-time

## Example 1: Buck Converter Design (12V → 5V)

```python
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *

circuit = Circuit('Buck Converter 12V to 5V')

# Define GaN HEMT model (simplified)
# IMPORTANT: 'lambda' is a Python keyword, use **kwargs: **{'lambda': 0.01}
circuit.model('gan_hemt', 'nmos', level=1, kp=500e-6, vto=1.5, **{'lambda': 0.01})

# Input voltage
circuit.V('in', 'Vin', circuit.gnd, 12@u_V)

# GaN HEMT switch (high-side in buck)
circuit.MOSFET('Q1', 'Vsw', 'Vgate', 'Vin', 'Vin', 
               model='gan_hemt', w=2000e-6, l=0.5e-6)

# Freewheeling diode (Schottky for low Vf)
# IMPORTANT: 'is' is a Python keyword, use **kwargs: **{'is': 1e-6}
circuit.model('schottky', 'd', **{'is': 1e-6}, rs=0.05, n=1.05)
circuit.D('D1', circuit.gnd, 'Vsw', model='schottky')

# Inductor (calculated: L = (12-5)*0.42/(0.4*10*500k) ≈ 7.4uH)
circuit.L('L1', 'Vsw', 'Vout', 10@u_uH)

# Output capacitor (for filtering)
circuit.C('C1', 'Vout', circuit.gnd, 100@u_uF)

# Load resistor (5V/10A = 0.5Ω for 50W)
circuit.R('load', 'Vout', circuit.gnd, 0.5@u_Ω)

# CRITICAL: Gate must be PWM, NOT DC!
# Calculate duty cycle with compensation for losses
Vin = 12
Vout_target = 5
I_load = 10 / 5  # 2A (P_out / V_out)
f_sw = 500e3  # 500kHz

# Calculate losses
V_diode = 0.5  # Schottky diode forward drop
R_ds_on = 0.01  # 10mΩ switch on-resistance
R_DCR = 0.02    # 20mΩ inductor DC resistance

V_switch_loss = R_ds_on * I_load  # 0.02V
V_inductor_loss = R_DCR * I_load  # 0.04V
V_loss_total = V_diode + V_switch_loss + V_inductor_loss  # 0.56V

# Compensated duty cycle
D_ideal = Vout_target / Vin  # 0.417 (41.7%)
D_compensated = (Vout_target + V_loss_total) / Vin  # 0.463 (46.3%)

period = 1 / f_sw  # 2µs
pulse_width = period * D_compensated  # 0.926µs

# PWM gate drive (PulseVoltageSource)
circuit.PulseVoltageSource('gate', 'Vgate', circuit.gnd,
                           initial_value=0@u_V, pulsed_value=5@u_V,
                           pulse_width=pulse_width@u_s, period=period@u_s,
                           delay_time=0@u_s, rise_time=1@u_ns, fall_time=1@u_ns)

simulator = circuit.simulator()
```

### Alternative: Voltage-Controlled Switch (VCS)
For more reliable switching behavior, use VCS instead of MOSFET model:

```python
# Replace MOSFET with voltage-controlled switch
# Switch ON when Vgate > 2.5V, OFF when < 2.5V
circuit.VCS('S1', 'Vsw', 'Vin', 'Vgate', circuit.gnd,
            model='ideal_switch')
circuit.model('ideal_switch', 'SW', ron=0.01, roff=1e6, von=2.5, voff=2.5)
```

VCS advantages:
- More reliable switching behavior in simulation
- Simpler than detailed MOSFET models
- Good for initial design validation

## Example 2: Boost Converter Design (5V → 12V)

```python
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *

circuit = Circuit('Boost Converter 5V to 12V')

# Input voltage
circuit.V('in', 'Vin', circuit.gnd, 5@u_V)

# LARGE inductor for boost (2-5× larger than buck)
circuit.L('L1', 'Vin', 'Vsw', 100@u_uH)

# MOSFET switch - LOW SIDE (between Vsw and GND for boost!)
circuit.model('NFET', 'nmos', level=1, vto=1.5, kp=200, **{'lambda': 0.001})
circuit.MOSFET('M1', 'Vsw', 'Vgate', circuit.gnd, circuit.gnd,
               model='NFET', w=10000e-6, l=0.5e-6)

# CRITICAL: Diode direction for boost
# Anode at Vsw (switching node), Cathode at Vout
# PySpice syntax: D(name, anode, cathode)
circuit.model('DMOD', 'D', **{'is': 1e-12}, rs=0.01, n=1.0)
circuit.D('D1', 'Vsw', 'Vout', model='DMOD')  # Vsw → Vout

# Output capacitor - MUST start at 0V for boost!
circuit.C('C1', 'Vout', circuit.gnd, 220@u_uF, initial_condition=0@u_V)

# Load resistor
circuit.R('load', 'Vout', circuit.gnd, 36@u_Ω)  # Light load helps startup

# PWM with loss compensation
Vin = 5.0
Vout_target = 12.0
f_sw = 300e3

# Boost duty cycle: D = 1 - (Vin/Vout)
D_ideal = 1 - (Vin / Vout_target)  # 0.583 (58.3%)

# Minimal compensation for boost (less sensitive than buck)
D_compensated = D_ideal * 1.0  # Can use 1.0-1.05 for boost

period = 1 / f_sw
pulse_width = period * D_compensated

circuit.PulseVoltageSource('gate', 'Vgate', circuit.gnd,
                           initial_value=0@u_V, pulsed_value=6@u_V,
                           pulse_width=pulse_width@u_s, period=period@u_s,
                           delay_time=1@u_us, rise_time=5@u_ns, fall_time=5@u_ns)

simulator = circuit.simulator()
# Note: Boost needs longer simulation time to ramp up from 0V
analysis = simulator.transient(step_time=100@u_ns, end_time=30@u_ms)
```

**Boost Converter Key Points:**
- Switch is LOW-SIDE (Vsw to GND), not high-side like buck
- Diode MUST be Vsw (anode) → Vout (cathode)
- Output cap starts at 0V, not pre-charged
- Larger inductor needed (100µH vs 20µH for buck)
- Longer simulation time to allow startup ramp
- Less sensitive to duty cycle compensation than buck

### Simulation Boilerplate Requirements (strict)

- Always import PySpice using:
  - `from PySpice.Spice.Netlist import Circuit`
  - `from PySpice.Unit import *`
- **Do NOT import from `PySpice.Spice.Simulation`** (e.g., `Transient`, `TransientAnalysis`). Simply call `simulator.transient(...)` on the circuit simulator instance.
- Avoid advanced helpers such as `ExportWaveForm`; use NumPy/matplotlib instead.
- If you need matplotlib, include `import matplotlib.pyplot as plt` and keep plotting optional (scripts may stub out `plt.show()`).

### Hard Requirements (LLM must obey)

- **Freewheeling path:** Every buck converter must include a diode (e.g., `circuit.D('D1', gnd, 'Vsw', model='DMOD')`) or a synchronous low-side switch so inductor current has a path when the main switch is off.
- **PWM gate drive:** Use `PulseVoltageSource` (or another explicit PWM source). **Never** use `SinusoidalVoltageSource`, DC sources, or behavioral expressions for gate drive.
- **Load computation:** Compute numeric component values (floats) before applying units. For example:
  ```python
  R_load_value = (Vout_target ** 2) / P_out
  circuit.R('load', 'Vout', circuit.gnd, R_load_value@u_Ω)
  ```
  Do **not** write expressions like `(Vout**2 / Power) @ u_Ω`.
- **Diode models:** Whenever you instantiate a diode, also create `circuit.model('<name>', 'D', ...)` with Schottky-like parameters and reference that model in the `circuit.D(...)` call.

#### Canonical Freewheel Snippet (copy this block)

```python
# Schottky diode model + connection (assumes switching node is 'Vsw')
circuit.model('DMOD', 'D', **{'is': 1e-9}, Rs=0.05, N=1.5)
circuit.D('D1', circuit.gnd, 'Vsw', model='DMOD')
```

Use the same naming if unsure; adjust node names only if your circuit uses different labels.

## Design Checklist

1. ✓ Topology selected based on Vin/Vout relationship
2. ✓ Inductor sized for acceptable ripple current
3. ✓ Capacitor sized for acceptable output ripple
4. ✓ Switching frequency chosen (balance size vs efficiency)
5. ✓ **PWM gate drive implemented (NOT DC voltage!)**
6. ✓ **Duty cycle calculated with loss compensation (diode + resistances)**
7. ✓ **Loss calculation includes: diode drop, switch Rds_on, inductor DCR**
8. ✓ GaN device or VCS properly modeled
8. ✓ Gate drive voltage appropriate (5-6V for GaN)
9. ✓ Freewheeling path provided (diode or synchronous switch)
10. ✓ Protection circuits considered

## Your Task

Design [TASK].

Specifications:
- Input voltage: [INPUT_VOLTAGE]V
- Output voltage: [OUTPUT_VOLTAGE]V  
- Output power: [POWER]W
- Switching frequency: [FREQ]kHz
- Efficiency target: [EFFICIENCY]%

Input nodes: [INPUT]
Output nodes: [OUTPUT]

## Answer

Provide:
1. Topology selection and reasoning
2. Component calculations
3. Complete PySpice code

## IMPORTANT: Python Keyword Conflicts

When using PySpice model() function, avoid Python keywords:
- **lambda** → Use: `**{'lambda': value}` instead of `lambda=value`
- **is** → Use: `**{'is': value}` instead of `is=value`

Example:
```python
# ❌ WRONG (causes SyntaxError):
circuit.model('mos', 'nmos', lambda=0.01, is=1e-6)

# ✅ CORRECT:
circuit.model('mos', 'nmos', **{'lambda': 0.01}, **{'is': 1e-6})
# OR combine:
circuit.model('mos', 'nmos', **{'lambda': 0.01, 'is': 1e-6})
```

