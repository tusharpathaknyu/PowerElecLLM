from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *
import matplotlib.pyplot as plt
import numpy as np

print("=== Fixed Boost Converter using Switch Model V2 ===\n")

circuit = Circuit('Boost Converter Fixed')

# Input voltage
circuit.V('in', 'Vin', circuit.gnd, 5@u_V)

# Inductor from Vin to switching node (larger for boost stability)
circuit.L('L1', 'Vin', 'Vsw', 47@u_uH)

# Use a simple resistor-based switch model instead of VCS
# When gate is HIGH: low resistance path to ground
# When gate is LOW: open circuit (high resistance)
# This avoids the VCS convergence issues

# Gate-controlled resistor (behavioral)
# We'll use a voltage-controlled voltage source to create switch behavior
circuit.V('switch_ctrl', 'Vgate_sense', circuit.gnd, 0@u_V)  # Sense gate voltage

# Better approach: Use subcircuit with controlled sources
# Actually, let's try NMOS with proper DC operating point

# Simple NMOS model for the switch with stronger parameters
circuit.model('NFET', 'nmos', level=1, vto=1.5, kp=100, **{'lambda': 0.001})
circuit.MOSFET('M1', 'Vsw', 'Vgate', circuit.gnd, circuit.gnd,
               model='NFET', w=5000e-6, l=0.5e-6)

# Diode from Vsw to Vout
circuit.model('DMOD', 'D', **{'is': 1e-12}, rs=0.05, n=1.05, bv=20)
circuit.D('D1', 'Vout', 'Vsw', model='DMOD')

# Output capacitor - MUST start at 0V or below Vin for boost to work
circuit.C('C1', 'Vout', circuit.gnd, 100@u_uF, initial_condition=0@u_V)

# Load
circuit.R('load', 'Vout', circuit.gnd, 7.2@u_Ω)

# Gate drive with ideal duty cycle
Vin = 5.0
Vout_target = 12.0
f_sw = 300e3

# Ideal D = 1 - Vin/Vout = 1 - 5/12 = 0.583
D_ideal = 1 - (Vin / Vout_target)
print(f"Ideal duty cycle: {D_ideal:.3f} ({D_ideal*100:.1f}%)")

period = 1 / f_sw
pulse_width = period * D_ideal

circuit.PulseVoltageSource(
    'gate', 'Vgate', circuit.gnd,
    initial_value=0@u_V,
    pulsed_value=6@u_V,  # Higher gate voltage for NMOS
    pulse_width=pulse_width@u_s,
    period=period@u_s,
    delay_time=0.5@u_us,  # Delay to let initial conditions settle
    rise_time=5@u_ns,
    fall_time=5@u_ns
)

try:
    simulator = circuit.simulator(temperature=25, nominal_temperature=25)
    # Longer simulation to allow boost to ramp up from 0V
    analysis = simulator.transient(step_time=50@u_ns, end_time=20@u_ms)
    
    time = np.array(analysis.time)
    vout = np.array(analysis['Vout'])
    vsw = np.array(analysis['Vsw'])
    vgate = np.array(analysis['Vgate'])
    
    # Analyze steady state
    steady_idx = int(len(vout) * 0.5)
    vout_final = vout[steady_idx:].mean()
    vout_ripple = vout[steady_idx:].max() - vout[steady_idx:].min()
    vsw_max = vsw[steady_idx:].max()
    vsw_min = vsw[steady_idx:].min()
    error_pct = abs(vout_final - 12) / 12 * 100
    
    print(f"\nResults:")
    print(f"  Vout: {vout_final:.3f}V (target: 12V)")
    print(f"  Error: {abs(vout_final-12)*1000:.1f}mV ({error_pct:.1f}%)")
    print(f"  Ripple: {vout_ripple*1000:.1f}mV")
    print(f"  Vsw range: {vsw_min:.2f}V to {vsw_max:.2f}V")
    
    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    time_ms = time * 1000
    
    # Vout
    axes[0].plot(time_ms, vout, 'b-', linewidth=2)
    axes[0].axhline(y=12, color='r', linestyle='--', linewidth=2, label='Target (12V)')
    axes[0].set_ylabel('Vout (V)')
    axes[0].set_title(f'Boost Converter Output - {vout_final:.3f}V ({error_pct:.1f}% error)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Vsw
    axes[1].plot(time_ms, vsw, 'g-', linewidth=1)
    axes[1].set_ylabel('Vsw (V)')
    axes[1].set_title('Switching Node Voltage')
    axes[1].grid(True, alpha=0.3)
    
    # Vgate
    axes[2].plot(time_ms, vgate, 'orange', linewidth=1)
    axes[2].set_xlabel('Time (ms)')
    axes[2].set_ylabel('Vgate (V)')
    axes[2].set_title('Gate Drive Signal')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
