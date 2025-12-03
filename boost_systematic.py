"""
Boost Converter Debug - Systematic approach
Let's verify each component step by step
"""
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *
import matplotlib.pyplot as plt
import numpy as np

print("="*60)
print("BOOST CONVERTER SYSTEMATIC DEBUG")
print("="*60)

# Start with minimal circuit - no load first
circuit = Circuit('Boost Minimal')

# 1. Input voltage
circuit.V('in', 'Vin', circuit.gnd, 5@u_V)
print("\n✓ Input: 5V")

# 2. Large inductor for better energy storage
# For boost: L_min = Vin * D * (1-D) / (f * ΔI)
# With D=0.58, f=300kHz, ΔI=0.5A: L > 40µH
circuit.L('L1', 'Vin', 'Vsw', 100@u_uH)
print("✓ Inductor: 100µH (large for stability)")

# 3. MOSFET switch - drain to Vsw, source to GND
# This is CRITICAL: for boost, switch must be between Vsw and GND
circuit.model('NFET', 'nmos', level=1, vto=1.5, kp=200, **{'lambda': 0.001})
circuit.MOSFET('M1', 'Vsw', 'Vgate', circuit.gnd, circuit.gnd,
               model='NFET', w=10000e-6, l=0.5e-6)
print("✓ MOSFET: Vsw → GND, controlled by Vgate")

# 4. Diode from Vsw to Vout (CRITICAL DIRECTION!)
# Anode at Vsw, Cathode at Vout
# PySpice: D(name, node+, node-) where node+ is anode
circuit.model('DMOD', 'D', **{'is': 1e-12}, rs=0.01, n=1.0, bv=30)
circuit.D('D1', 'Vsw', 'Vout', model='DMOD')  # FIXED: Vsw (anode) first!
print("✓ Diode: Vsw (anode) → Vout (cathode)")

# 5. Large output capacitor
circuit.C('C1', 'Vout', circuit.gnd, 220@u_uF, initial_condition=0@u_V)
print("✓ Cap: 220µF starting at 0V")

# 6. Light load initially to help startup
# Start with 36Ω (0.33A at 12V = 4W)
circuit.R('load', 'Vout', circuit.gnd, 36@u_Ω)
print("✓ Load: 36Ω (light load for startup)")

# 7. PWM gate drive
Vin_val = 5.0
Vout_target = 12.0
f_sw = 300e3

# Ideal duty cycle for boost: D = 1 - (Vin/Vout)
D_ideal = 1 - (Vin_val / Vout_target)
print(f"\n✓ Duty cycle: {D_ideal:.3f} ({D_ideal*100:.1f}%)")

period = 1 / f_sw
pulse_width = period * D_ideal

circuit.PulseVoltageSource(
    'gate', 'Vgate', circuit.gnd,
    initial_value=0@u_V,
    pulsed_value=6@u_V,
    pulse_width=pulse_width@u_s,
    period=period@u_s,
    delay_time=1@u_us,  # Small startup delay
    rise_time=5@u_ns,
    fall_time=5@u_ns
)

print("\n" + "="*60)
print("RUNNING SIMULATION (30ms to allow ramp-up)")
print("="*60 + "\n")

try:
    simulator = circuit.simulator(temperature=25, nominal_temperature=25)
    analysis = simulator.transient(step_time=100@u_ns, end_time=30@u_ms)
    
    time = np.array(analysis.time) * 1000  # Convert to ms
    vout = np.array(analysis['Vout'])
    vsw = np.array(analysis['Vsw'])
    vgate = np.array(analysis['Vgate'])
    vin = np.array(analysis['Vin'])
    
    # Also get inductor current
    try:
        il = np.array(analysis['vl1'])  # Inductor voltage
    except:
        il = None
    
    # Analyze final state (last 20%)
    steady_idx = int(len(vout) * 0.8)
    vout_final = vout[steady_idx:].mean()
    vout_ripple = vout[steady_idx:].max() - vout[steady_idx:].min()
    vsw_max = vsw[steady_idx:].max()
    vsw_min = vsw[steady_idx:].min()
    
    error_v = vout_final - Vout_target
    error_pct = abs(error_v) / Vout_target * 100
    
    print(f"RESULTS:")
    print(f"  Target: {Vout_target}V")
    print(f"  Actual: {vout_final:.3f}V")
    print(f"  Error: {error_v:+.3f}V ({error_pct:.1f}%)")
    print(f"  Ripple: {vout_ripple*1000:.1f}mV")
    print(f"  Vsw range: {vsw_min:.2f}V to {vsw_max:.2f}V")
    
    # Check if Vsw is reaching above Vout (necessary for boost)
    if vsw_max > vout_final:
        print(f"  ✓ Vsw exceeds Vout ({vsw_max:.1f}V > {vout_final:.1f}V) - boost action working")
    else:
        print(f"  ✗ Vsw NOT exceeding Vout - circuit not boosting!")
    
    # Plot detailed results
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(4, 1, hspace=0.3)
    
    # Output voltage
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(time, vout, 'b-', linewidth=2, label='Vout')
    ax1.axhline(y=Vout_target, color='r', linestyle='--', linewidth=2, label=f'Target ({Vout_target}V)')
    ax1.axhline(y=Vin_val, color='g', linestyle=':', linewidth=1, label=f'Vin ({Vin_val}V)')
    ax1.set_ylabel('Vout (V)')
    ax1.set_title(f'Boost Converter Output: {vout_final:.3f}V ({error_pct:.1f}% error)')
    ax1.legend(loc='right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([-1, 15])
    
    # Switching node
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(time, vsw, 'g-', linewidth=1, label='Vsw')
    ax2.axhline(y=vout_final, color='b', linestyle='--', linewidth=1, label=f'Vout ({vout_final:.1f}V)')
    ax2.set_ylabel('Vsw (V)')
    ax2.set_title('Switching Node Voltage')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Gate drive
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(time, vgate, 'orange', linewidth=1, label='Vgate')
    ax3.set_ylabel('Vgate (V)')
    ax3.set_title(f'Gate Drive ({D_ideal*100:.1f}% duty cycle)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Zoom on last 0.5ms to see switching detail
    ax4 = fig.add_subplot(gs[3])
    zoom_start = -0.1  # Last 100µs
    zoom_idx = np.where(time >= time[-1] + zoom_start)[0][0]
    ax4.plot(time[zoom_idx:], vsw[zoom_idx:], 'g-', linewidth=1.5, label='Vsw')
    ax4.plot(time[zoom_idx:], vout[zoom_idx:], 'b-', linewidth=1.5, label='Vout')
    ax4.set_xlabel('Time (ms)')
    ax4.set_ylabel('Voltage (V)')
    ax4.set_title('Zoomed: Last 100µs of Switching')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.savefig('/Users/tushardhananjaypathak/Desktop/PowerElecLLM/boost_debug.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved: boost_debug.png")
    plt.show()
    
    # Success criteria
    print("\n" + "="*60)
    if abs(error_pct) < 10:
        print("✅ SUCCESS! Boost converter working within 10% tolerance")
    else:
        print("❌ FAILED! Output voltage not reaching target")
        print("\nDiagnostic suggestions:")
        if vout_final < 1:
            print("  - Output stuck near 0V: check diode direction")
            print("  - Verify MOSFET is switching properly")
        elif vout_final < Vin_val:
            print("  - Output below input: boost action not working")
            print("  - Check Vsw is rising above Vout when switch opens")
        else:
            print("  - Adjust duty cycle compensation")
    print("="*60)
    
except Exception as e:
    print(f"\n❌ SIMULATION ERROR:")
    print(f"   {e}")
    import traceback
    traceback.print_exc()
