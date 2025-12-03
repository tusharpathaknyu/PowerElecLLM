from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *
import matplotlib.pyplot as plt
import numpy as np

# Test 1: Simple boost converter with minimal components
print("=== Testing Boost Converter Circuit ===\n")

circuit = Circuit('Boost Test')

# Input voltage
circuit.V('in', 'Vin', circuit.gnd, 5@u_V)

# Inductor
circuit.L('L1', 'Vin', 'Vsw', 22@u_uH)

# Switch (between Vsw and GND)
circuit.VCS('SW1', 'Vsw', circuit.gnd, 'Vgate', circuit.gnd, model='SMOD')
circuit.model('SMOD', 'SW', Ron=0.01, Roff=1e6, Vt=2.5, Vh=0.5)

# Diode (from Vsw to Vout)
circuit.model('DMOD', 'D', **{'is': 1e-9}, Rs=0.05, N=1.8)
circuit.D('D1', 'Vout', 'Vsw', model='DMOD')

# Output cap (start with small initial voltage to avoid issues)
circuit.C('C1', 'Vout', circuit.gnd, 47@u_uF, initial_condition=5@u_V)

# Load
circuit.R('load', 'Vout', circuit.gnd, 7.2@u_Ω)

# Gate drive - test different duty cycles
test_duty_cycles = [0.50, 0.55, 0.60, 0.65]

for D_test in test_duty_cycles:
    print(f"\n--- Testing D = {D_test:.2f} ({D_test*100:.0f}%) ---")
    
    # Remove old gate source if exists
    try:
        circuit._elements.pop('Vgate', None)
    except:
        pass
    
    f_sw = 300e3
    period = 1 / f_sw
    pulse_width = period * D_test
    
    circuit.PulseVoltageSource(
        'gate', 'Vgate', circuit.gnd,
        initial_value=0@u_V,
        pulsed_value=5@u_V,
        pulse_width=pulse_width@u_s,
        period=period@u_s,
        delay_time=0.1@u_us,  # Small delay to let circuit settle
        rise_time=10@u_ns,
        fall_time=10@u_ns
    )
    
    try:
        simulator = circuit.simulator(temperature=25, nominal_temperature=25)
        
        analysis = simulator.transient(
            step_time=50@u_ns,
            end_time=5@u_ms
        )
        
        # Get final values
        time = np.array(analysis.time)
        vout = np.array(analysis['Vout'])
        vsw = np.array(analysis['Vsw'])
        
        # Check steady state (last 30%)
        steady_idx = int(len(vout) * 0.7)
        vout_final = vout[steady_idx:].mean()
        vsw_max = vsw[steady_idx:].max()
        vsw_min = vsw[steady_idx:].min()
        
        print(f"  Vout (final): {vout_final:.3f}V")
        print(f"  Vsw range: {vsw_min:.1f}V to {vsw_max:.1f}V")
        print(f"  Error from 12V target: {abs(vout_final - 12):.3f}V ({abs(vout_final-12)/12*100:.1f}%)")
        
        # Check if this is the best one
        if abs(vout_final - 12) < 0.6:  # Within 5%
            print(f"  ✓ GOOD DUTY CYCLE!")
            
            # Plot this one
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            time_ms = time * 1000
            ax1.plot(time_ms, vout, 'b-', linewidth=2)
            ax1.axhline(y=12, color='r', linestyle='--', linewidth=2)
            ax1.set_xlabel('Time (ms)')
            ax1.set_ylabel('Vout (V)')
            ax1.set_title(f'Boost Converter with D={D_test:.2f} - Vout={vout_final:.3f}V')
            ax1.grid(True, alpha=0.3)
            
            ax2.plot(time_ms, vsw, 'g-', linewidth=1)
            ax2.set_xlabel('Time (ms)')
            ax2.set_ylabel('Vsw (V)')
            ax2.set_title('Switching Node')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'/Users/tushardhananjaypathak/Desktop/PowerElecLLM/boost_D{int(D_test*100)}.png', dpi=150)
            print(f"  Plot saved: boost_D{int(D_test*100)}.png")
            
    except Exception as e:
        print(f"  ERROR: {str(e)[:100]}")

print("\n=== Testing Complete ===")
