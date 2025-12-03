from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *

import matplotlib.pyplot as plt
import numpy as np

"""Reference test: 24V -> 12V buck converter, 50W, 500kHz"""

Vin = 24.0
Vout_target = 12.0
P_out = 50.0  # Watts
I_load = P_out / Vout_target  # 4.17A
f_switching = 500e3  # 500kHz

circuit = Circuit('Reference Buck 24V to 12V')

# Input supply
circuit.V('in', 'Vin', circuit.gnd, Vin@u_V)

# Use VCS for reliable switching
circuit.VCS('SW1', 'Vin', 'Vsw', 'Vgate', circuit.gnd, model='SWITCH')
circuit.model('SWITCH', 'SW', Ron=0.05, Roff=1@u_MΩ, Vt=2.5, Vh=0.5)

# Freewheel diode
circuit.model('DMOD', 'D', **{'is': 1e-9}, Rs=0.02, N=1.6)
circuit.D('D1', circuit.gnd, 'Vsw', model='DMOD')

# Inductor sizing (~15µH)
circuit.L('L1', 'Vsw', 'Vout', 15@u_uH)

# Output capacitor (47µF) with zero initial voltage
circuit.C('C1', 'Vout', circuit.gnd, 47@u_uF, initial_condition=0@u_V)

# Load resistor (≈ 2.88Ω)
R_load = (Vout_target ** 2) / P_out
circuit.R('load', 'Vout', circuit.gnd, R_load@u_Ω)

# Loss estimation
V_diode = 0.4
R_switch_on = 0.05
R_inductor_DCR = 0.015

V_switch_loss = R_switch_on * I_load
V_inductor_loss = R_inductor_DCR * I_load
V_loss_total = V_diode + V_switch_loss + V_inductor_loss

D_ideal = Vout_target / Vin
D_compensated = min(0.95, (Vout_target + V_loss_total) / Vin)

period = 1 / f_switching
pulse_width = period * D_compensated

print("=== Buck 24V->12V Duty Cycle ===")
print(f"Load current: {I_load:.2f}A")
print(f"Diode drop: {V_diode:.3f}V, Switch loss: {V_switch_loss:.3f}V, Inductor loss: {V_inductor_loss:.3f}V")
print(f"Ideal D: {D_ideal:.3f} ({D_ideal*100:.1f}%)")
print(f"Compensated D: {D_compensated:.3f} ({D_compensated*100:.1f}%)")

circuit.PulseVoltageSource('gate', 'Vgate', circuit.gnd,
                           initial_value=0@u_V, pulsed_value=5@u_V,
                           pulse_width=pulse_width@u_s, period=period@u_s,
                           delay_time=0@u_us, rise_time=5@u_ns, fall_time=5@u_ns)

simulator = circuit.simulator(temperature=25, nominal_temperature=25)
analysis = simulator.transient(step_time=50@u_ns, end_time=8@u_ms)

time_ms = np.array(analysis.time) * 1000
vout = np.array(analysis['Vout'])
vsw = np.array(analysis['Vsw'])

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
ax1.plot(time_ms, vout, 'b-', linewidth=2, label='Vout')
ax1.axhline(Vout_target, color='r', linestyle='--', label=f'Target ({Vout_target}V)')
ax1.set_title('Buck Converter Output (24V → 12V, 50W, 500kHz)')
ax1.set_xlabel('Time (ms)')
ax1.set_ylabel('Vout (V)')
ax1.grid(True, alpha=0.3)
ax1.legend()

ax2.plot(time_ms, vsw, 'g-', linewidth=1, label='Vsw')
ax2.set_title('Switching Node Voltage')
ax2.set_xlabel('Time (ms)')
ax2.set_ylabel('Voltage (V)')
ax2.grid(True, alpha=0.3)
ax2.legend()

steady = vout[int(len(vout)*0.7):]
vout_final = float(np.mean(steady))
ripple = float(np.max(steady) - np.min(steady))
error = vout_final - Vout_target

stats = f'Final: {vout_final:.3f}V\nRipple: {ripple*1000:.1f}mV\nError: {error*1000:.1f}mV ({error/Vout_target*100:.1f}%)'
ax1.text(0.02, 0.98, stats, transform=ax1.transAxes,
         va='top', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))

plt.tight_layout()
plt.show()

print("\n=== Simulation Summary ===")
print(f"Average Vout: {vout_final:.3f}V")
print(f"Error: {error*1000:.1f} mV ({error/Vout_target*100:.1f}%)")
print(f"Ripple: {ripple*1000:.1f} mV")
print(f"Switching node range: {np.min(vsw):.2f}V – {np.max(vsw):.2f}V")
print(f"Duty cycle: {D_compensated*100:.1f}% @ {f_switching/1e3:.0f} kHz")
