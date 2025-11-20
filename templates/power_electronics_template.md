You aim to design a power electronic circuit topology for a given specification.
Please ensure your designed circuit topology works properly and achieves the design requirements.

## Power Electronics Design Principles

### 1. Topology Selection
Choose the appropriate converter topology based on requirements:
- **Buck Converter**: Step-down (Vin > Vout), non-isolated
- **Boost Converter**: Step-up (Vin < Vout), non-isolated  
- **Buck-Boost**: Bidirectional, non-isolated
- **Flyback**: Isolated, low-medium power (<100W)
- **Forward**: Isolated, medium power
- **LLC Resonant**: High efficiency, high power

### 2. Component Sizing Formulas

**Inductor (L)**:
- Buck: L = (Vin - Vout) × D / (ΔI_L × f_sw)
- Boost: L = Vin × D / (ΔI_L × f_sw)
- Where: D = duty cycle, ΔI_L = inductor ripple current (20-40% of I_out), f_sw = switching frequency

**Output Capacitor (C)**:
- C = I_out × D / (ΔV_out × f_sw)
- Where: ΔV_out = output voltage ripple (typically <1% of Vout)

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

## Example: Buck Converter Design

```python
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *

circuit = Circuit('Buck Converter 12V to 5V')

# Define GaN HEMT model (simplified)
circuit.model('gan_hemt', 'nmos', level=1, kp=500e-6, vto=1.5, lambda=0.01)

# Input voltage
circuit.V('in', 'Vin', circuit.gnd, 12@u_V)

# GaN HEMT switch (high-side in buck)
circuit.MOSFET('Q1', 'Vsw', 'Vgate', 'Vin', 'Vin', 
               model='gan_hemt', w=2000e-6, l=0.5e-6)

# Freewheeling diode (Schottky for low Vf)
circuit.model('schottky', 'd', is=1e-6, rs=0.05, n=1.05)
circuit.D('D1', circuit.gnd, 'Vsw', model='schottky')

# Inductor (calculated: L = (12-5)*0.42/(0.4*10*500k) ≈ 7.4uH)
circuit.L('L1', 'Vsw', 'Vout', 10@u_uH)

# Output capacitor (for filtering)
circuit.C('C1', 'Vout', circuit.gnd, 100@u_uF)

# Load resistor (5V/10A = 0.5Ω for 50W)
circuit.R('load', 'Vout', circuit.gnd, 0.5@u_Ω)

# Gate drive voltage (PWM signal - shown as DC for initial design)
circuit.V('gate', 'Vgate', circuit.gnd, 5@u_V)

simulator = circuit.simulator()
```

## Design Checklist

1. ✓ Topology selected based on Vin/Vout relationship
2. ✓ Inductor sized for acceptable ripple current
3. ✓ Capacitor sized for acceptable output ripple
4. ✓ Switching frequency chosen (balance size vs efficiency)
5. ✓ GaN device properly modeled
6. ✓ Gate drive voltage appropriate (5-6V for GaN)
7. ✓ Freewheeling path provided (diode or synchronous switch)
8. ✓ Protection circuits considered

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

