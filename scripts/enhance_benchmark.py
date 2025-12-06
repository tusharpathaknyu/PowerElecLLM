#!/usr/bin/env python3
"""
Comprehensive Benchmark Enhancement Script
==========================================
1. Add more topologies (Level 5 - Expert)
2. Create harder multi-stage problems
3. Generate bigger fine-tuning dataset
"""

import json
import os
import random
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).parent.parent


# ============================================================================
# PART 1: NEW TOPOLOGIES - LEVEL 5 EXPERT PROBLEMS
# ============================================================================

def generate_level5_problems() -> List[Dict]:
    """Generate expert-level problems with advanced topologies and constraints."""
    problems = []
    problem_id = 1
    
    # 1. Resonant Converters (LLC, Series Resonant)
    llc_problems = [
        {
            "id": f"L5_{problem_id:03d}",
            "prompt": "Design an LLC resonant converter for a 400V DC bus to 48V/20A server power supply. Target efficiency >96%, switching frequency 100-200kHz. Specify Lr, Lm, Cr, and turns ratio.",
            "specs": {"topology": "llc_resonant", "vin": 400, "vout": 48, "iout": 20, "power": 960, "efficiency_target": 0.96, "fsw_range": [100e3, 200e3]},
            "constraints": {"zcs": True, "zvs": True, "hold_up_time_ms": 20},
            "difficulty": "expert"
        },
        {
            "id": f"L5_{problem_id+1:03d}",
            "prompt": "Design an LLC converter for EV onboard charger: 800V battery pack, 11kW power, bidirectional operation. Include resonant tank design.",
            "specs": {"topology": "llc_resonant", "vin": 400, "vout": 800, "iout": 13.75, "power": 11000, "bidirectional": True},
            "constraints": {"zvs_range": [0.2, 1.0], "frequency_range": [50e3, 150e3]},
            "difficulty": "expert"
        },
    ]
    problems.extend(llc_problems)
    problem_id += 2
    
    # 2. Multi-phase Interleaved Converters
    multiphase_problems = [
        {
            "id": f"L5_{problem_id:03d}",
            "prompt": "Design a 4-phase interleaved buck converter for CPU VRM: 12V to 1.2V at 150A. Calculate phase inductors, output capacitance, and phase shift angles for minimum ripple.",
            "specs": {"topology": "multiphase_buck", "vin": 12, "vout": 1.2, "iout": 150, "power": 180, "num_phases": 4, "fsw": 500e3},
            "constraints": {"vout_ripple_pct": 1, "transient_response_us": 10, "current_sharing_error_pct": 5},
            "difficulty": "expert"
        },
        {
            "id": f"L5_{problem_id+1:03d}",
            "prompt": "Design a 3-phase interleaved boost PFC for 3kW telecom rectifier: 85-265VAC input, 400VDC output. Specify per-phase inductance and switching strategy.",
            "specs": {"topology": "multiphase_boost_pfc", "vin_range": [85, 265], "vout": 400, "power": 3000, "num_phases": 3, "pf_target": 0.99},
            "constraints": {"thd_pct": 5, "efficiency_target": 0.98},
            "difficulty": "expert"
        },
    ]
    problems.extend(multiphase_problems)
    problem_id += 2
    
    # 3. Cascaded/Multi-stage Converters
    cascaded_problems = [
        {
            "id": f"L5_{problem_id:03d}",
            "prompt": "Design a two-stage converter (PFC + LLC) for a 1kW LED driver: 90-264VAC input, 54V/18A output. Optimize for efficiency and power density.",
            "specs": {"topology": "pfc_llc_cascade", "vin_ac_range": [90, 264], "vout": 54, "iout": 18, "power": 972},
            "constraints": {"pf": 0.95, "efficiency_target": 0.93, "dc_link_voltage": 400},
            "difficulty": "expert"
        },
        {
            "id": f"L5_{problem_id+1:03d}",
            "prompt": "Design a cascaded buck-boost for solar MPPT: 18-45V input (panel voltage), 48V battery output, 500W. Handle wide input range efficiently.",
            "specs": {"topology": "cascaded_buck_boost", "vin_range": [18, 45], "vout": 48, "power": 500},
            "constraints": {"mppt_efficiency": 0.99, "battery_ripple_pct": 2},
            "difficulty": "expert"
        },
    ]
    problems.extend(cascaded_problems)
    problem_id += 2
    
    # 4. Soft-switching / ZVS Converters
    zvs_problems = [
        {
            "id": f"L5_{problem_id:03d}",
            "prompt": "Design a phase-shifted full-bridge (PSFB) converter with ZVS: 400V to 28V/50A for industrial power supply. Calculate leakage inductance and dead-time for ZVS.",
            "specs": {"topology": "phase_shifted_full_bridge", "vin": 400, "vout": 28, "iout": 50, "power": 1400, "fsw": 100e3},
            "constraints": {"zvs_range": [0.1, 1.0], "duty_loss_pct": 5},
            "difficulty": "expert"
        },
        {
            "id": f"L5_{problem_id+1:03d}",
            "prompt": "Design an active-clamp forward converter with ZVS for 48V telecom: 36-75V input, 3.3V/30A output. Specify clamp capacitor and magnetizing inductance.",
            "specs": {"topology": "active_clamp_forward", "vin_range": [36, 75], "vout": 3.3, "iout": 30, "power": 99, "fsw": 250e3},
            "constraints": {"zvs": True, "reset_voltage_margin": 0.9},
            "difficulty": "expert"
        },
    ]
    problems.extend(zvs_problems)
    problem_id += 2
    
    # 5. High Step-up/Step-down Ratio
    extreme_ratio_problems = [
        {
            "id": f"L5_{problem_id:03d}",
            "prompt": "Design a coupled-inductor boost converter for fuel cell: 20-40V input to 400V output (10:1 to 20:1 ratio), 2kW. Address extreme duty cycle limitations.",
            "specs": {"topology": "coupled_inductor_boost", "vin_range": [20, 40], "vout": 400, "power": 2000},
            "constraints": {"turns_ratio_range": [2, 4], "duty_max": 0.7},
            "difficulty": "expert"
        },
        {
            "id": f"L5_{problem_id+1:03d}",
            "prompt": "Design a switched-capacitor converter for 48V to 1V point-of-load with 100A capability. Compare against buck for this extreme ratio.",
            "specs": {"topology": "switched_capacitor", "vin": 48, "vout": 1, "iout": 100, "power": 100},
            "constraints": {"soft_charging": True, "num_stages": 4},
            "difficulty": "expert"
        },
    ]
    problems.extend(extreme_ratio_problems)
    problem_id += 2
    
    # 6. Bidirectional Converters
    bidirectional_problems = [
        {
            "id": f"L5_{problem_id:03d}",
            "prompt": "Design a dual-active-bridge (DAB) converter for V2G application: 400V battery to 800V DC bus, 10kW bidirectional. Calculate transformer turns ratio and phase shift control range.",
            "specs": {"topology": "dual_active_bridge", "v1": 400, "v2": 800, "power": 10000, "bidirectional": True, "fsw": 50e3},
            "constraints": {"zvs_both_directions": True, "efficiency_target": 0.97},
            "difficulty": "expert"
        },
        {
            "id": f"L5_{problem_id+1:03d}",
            "prompt": "Design a bidirectional interleaved buck-boost for 48V/400V hybrid energy storage: 5kW, seamless mode transition. Specify control strategy.",
            "specs": {"topology": "bidirectional_interleaved", "v_low": 48, "v_high": 400, "power": 5000, "num_phases": 2},
            "constraints": {"mode_transition_time_us": 100, "current_ripple_pct": 20},
            "difficulty": "expert"
        },
    ]
    problems.extend(bidirectional_problems)
    problem_id += 2
    
    # 7. GaN/SiC High-frequency Design
    wbg_problems = [
        {
            "id": f"L5_{problem_id:03d}",
            "prompt": "Design a GaN-based totem-pole PFC at 1MHz switching: 230VAC to 400VDC, 3kW. Account for GaN-specific layout and dead-time requirements.",
            "specs": {"topology": "totem_pole_pfc", "vin_ac": 230, "vout": 400, "power": 3000, "fsw": 1e6, "switch_type": "GaN"},
            "constraints": {"dead_time_ns": 20, "layout_loop_inductance_nh": 1, "pf": 0.99},
            "difficulty": "expert"
        },
        {
            "id": f"L5_{problem_id+1:03d}",
            "prompt": "Design a SiC-based 3-level flying capacitor converter for solar inverter: 1500V DC link, 500kW, grid-tied. Calculate flying capacitor values and balancing strategy.",
            "specs": {"topology": "flying_capacitor_3level", "vdc": 1500, "power": 500000, "fsw": 20e3, "switch_type": "SiC"},
            "constraints": {"voltage_balancing_error_pct": 2, "thd_pct": 3},
            "difficulty": "expert"
        },
    ]
    problems.extend(wbg_problems)
    problem_id += 2
    
    # 8. Matrix Converters
    matrix_problems = [
        {
            "id": f"L5_{problem_id:03d}",
            "prompt": "Design a 3x3 matrix converter for variable speed drive: 400VAC 50Hz input, 0-400VAC 0-100Hz output, 15kW. Specify modulation strategy and commutation sequence.",
            "specs": {"topology": "matrix_converter_3x3", "vin_ac": 400, "fin": 50, "vout_max": 400, "fout_range": [0, 100], "power": 15000},
            "constraints": {"input_pf": 1, "output_voltage_transfer_ratio": 0.866},
            "difficulty": "expert"
        },
    ]
    problems.extend(matrix_problems)
    problem_id += 1
    
    # 9. Digital Control Challenges
    control_problems = [
        {
            "id": f"L5_{problem_id:03d}",
            "prompt": "Design voltage-mode control loop for a buck converter: Vin=12V, Vout=3.3V, Iout=10A, Fsw=500kHz. Calculate Type-III compensator values for 50kHz crossover with 60° phase margin.",
            "specs": {"topology": "buck_vmc", "vin": 12, "vout": 3.3, "iout": 10, "fsw": 500e3, "L": 2.2e-6, "C": 100e-6, "esr": 5e-3},
            "constraints": {"crossover_freq": 50e3, "phase_margin_deg": 60, "gain_margin_db": 10},
            "difficulty": "expert"
        },
        {
            "id": f"L5_{problem_id+1:03d}",
            "prompt": "Design peak current mode control for a flyback converter in DCM: Vin=85-265VAC, Vout=24V, Pout=60W. Calculate slope compensation and current sense resistor.",
            "specs": {"topology": "flyback_pcmc", "vin_range": [85, 265], "vout": 24, "power": 60, "fsw": 100e3},
            "constraints": {"slope_compensation_ratio": 0.5, "subharmonic_immunity": True},
            "difficulty": "expert"
        },
    ]
    problems.extend(control_problems)
    problem_id += 2
    
    # 10. Thermal/Reliability Constraints
    thermal_problems = [
        {
            "id": f"L5_{problem_id:03d}",
            "prompt": "Design a 1kW buck converter with thermal constraints: junction temp <125°C at 85°C ambient, natural convection cooling only. Select MOSFETs and calculate heatsink requirements.",
            "specs": {"topology": "buck_thermal", "vin": 48, "vout": 12, "power": 1000, "t_ambient": 85, "t_junction_max": 125},
            "constraints": {"cooling": "natural_convection", "rth_heatsink_max": 2.5},
            "difficulty": "expert"
        },
    ]
    problems.extend(thermal_problems)
    
    return problems


# ============================================================================
# PART 2: HARDER MULTI-CONSTRAINT PROBLEMS (Level 4+)
# ============================================================================

def generate_harder_l4_problems() -> List[Dict]:
    """Generate harder Level 4 problems with multiple constraints."""
    problems = []
    
    # Multi-constraint optimization problems
    multi_constraint = [
        {
            "id": "L4_201",
            "prompt": "Design a buck converter for automotive 12V→5V, 3A with: <1% output ripple, >92% efficiency, <100kHz EMI compliance, -40°C to 85°C operation. Specify all component values and justify selections.",
            "specs": {"topology": "buck", "vin": 12, "vout": 5, "iout": 3, "power": 15},
            "constraints": {"vout_ripple_pct": 1, "efficiency_min": 0.92, "fsw_max": 100e3, "temp_range": [-40, 85]},
            "difficulty": "advanced"
        },
        {
            "id": "L4_202",
            "prompt": "Design a boost converter for solar battery charger: 12-18V panel to 24V battery, 100W. Must handle MPPT tracking with <0.5% tracking error and provide input current limiting at 7A.",
            "specs": {"topology": "boost", "vin_range": [12, 18], "vout": 24, "power": 100},
            "constraints": {"mppt_error_pct": 0.5, "iin_limit": 7, "soft_start_time_ms": 50},
            "difficulty": "advanced"
        },
        {
            "id": "L4_203",
            "prompt": "Design a flyback for isolated 48V→5V/10A auxiliary supply with: 5000V isolation, <50mV ripple, creepage >8mm. Include transformer design with specific core selection.",
            "specs": {"topology": "flyback", "vin": 48, "vout": 5, "iout": 10, "power": 50},
            "constraints": {"isolation_voltage": 5000, "vout_ripple_mv": 50, "creepage_mm": 8},
            "difficulty": "advanced"
        },
        {
            "id": "L4_204",
            "prompt": "Design a full-bridge converter for 48V to ±15V dual output, 200W total (symmetric loads). Both outputs must track within 1% and share a single transformer.",
            "specs": {"topology": "full_bridge_dual_output", "vin": 48, "vout_pos": 15, "vout_neg": -15, "power": 200},
            "constraints": {"output_tracking_pct": 1, "cross_regulation_pct": 3},
            "difficulty": "advanced"
        },
        {
            "id": "L4_205",
            "prompt": "Design a SEPIC for wide-input LED driver: 9-36V DC input to constant 700mA output (LED forward voltage 24-32V depending on temp). Implement constant current control.",
            "specs": {"topology": "sepic", "vin_range": [9, 36], "iout": 0.7, "vout_range": [24, 32]},
            "constraints": {"iout_regulation_pct": 2, "cc_mode": True, "dimming_range": [0.1, 1.0]},
            "difficulty": "advanced"
        },
    ]
    problems.extend(multi_constraint)
    
    # Efficiency optimization at light/heavy load
    efficiency_problems = [
        {
            "id": "L4_206",
            "prompt": "Design a buck converter optimized for 10%-100% load efficiency: 24V→5V, 0.5-5A. Achieve >85% efficiency at 10% load and >93% at full load. Consider PFM/PWM mode switching.",
            "specs": {"topology": "buck", "vin": 24, "vout": 5, "iout_range": [0.5, 5]},
            "constraints": {"efficiency_10pct_load": 0.85, "efficiency_full_load": 0.93, "mode_switch": "PFM_PWM"},
            "difficulty": "advanced"
        },
        {
            "id": "L4_207",
            "prompt": "Design a boost converter for portable device: 3.0-4.2V (1S Li-ion) to 5V, 2A max. Optimize for <1mA quiescent current at no load while maintaining regulation.",
            "specs": {"topology": "boost", "vin_range": [3.0, 4.2], "vout": 5, "iout": 2},
            "constraints": {"iq_ua": 1000, "vout_regulation_pct": 2},
            "difficulty": "advanced"
        },
    ]
    problems.extend(efficiency_problems)
    
    # Transient response challenges
    transient_problems = [
        {
            "id": "L4_208",
            "prompt": "Design a buck converter with fast transient response: 12V→1.8V, 0-20A load step. Must recover to within 3% in <50µs. Calculate output capacitance and control bandwidth.",
            "specs": {"topology": "buck", "vin": 12, "vout": 1.8, "iout_step": 20},
            "constraints": {"settling_time_us": 50, "overshoot_pct": 3, "undershoot_pct": 3},
            "difficulty": "advanced"
        },
        {
            "id": "L4_209",
            "prompt": "Design a boost converter with input voltage transient immunity: 4.5-5.5V input, 12V output, 3A. Must maintain output within 5% during 1V/µs input slew rate.",
            "specs": {"topology": "boost", "vin_range": [4.5, 5.5], "vout": 12, "iout": 3},
            "constraints": {"vin_slew_rate_v_per_us": 1, "vout_deviation_pct": 5},
            "difficulty": "advanced"
        },
    ]
    problems.extend(transient_problems)
    
    # Size/cost constrained
    size_cost_problems = [
        {
            "id": "L4_210",
            "prompt": "Design a buck converter fitting in 10x10mm: 5V→3.3V, 2A. Minimize total inductor+capacitor volume while meeting 20mV ripple spec. Switching frequency is your optimization variable.",
            "specs": {"topology": "buck", "vin": 5, "vout": 3.3, "iout": 2},
            "constraints": {"board_area_mm2": 100, "vout_ripple_mv": 20, "height_mm": 3},
            "difficulty": "advanced"
        },
    ]
    problems.extend(size_cost_problems)
    
    return problems


# ============================================================================
# PART 3: GENERATE BIGGER FINE-TUNING DATASET
# ============================================================================

def generate_large_finetune_dataset(num_examples: int = 2000) -> List[Dict]:
    """Generate a large fine-tuning dataset with variations."""
    examples = []
    
    # Define parameter ranges for each topology
    topologies = {
        "buck": {
            "vin_range": [5, 100],
            "vout_ratio_range": [0.1, 0.9],  # vout/vin
            "power_range": [1, 1000],
        },
        "boost": {
            "vin_range": [3, 48],
            "vout_ratio_range": [1.1, 5.0],  # vout/vin
            "power_range": [1, 500],
        },
        "buck_boost": {
            "vin_range": [5, 48],
            "vout_range": [3, 48],
            "power_range": [1, 200],
        },
        "sepic": {
            "vin_range": [5, 36],
            "vout_range": [3, 24],
            "power_range": [1, 100],
        },
        "cuk": {
            "vin_range": [5, 36],
            "vout_range": [-24, -3],
            "power_range": [1, 100],
        },
        "flyback": {
            "vin_range": [12, 400],
            "vout_range": [3.3, 48],
            "power_range": [5, 150],
        },
        "forward": {
            "vin_range": [24, 400],
            "vout_range": [5, 48],
            "power_range": [10, 500],
        },
        "half_bridge": {
            "vin_range": [100, 400],
            "vout_range": [12, 48],
            "power_range": [50, 1000],
        },
        "full_bridge": {
            "vin_range": [200, 800],
            "vout_range": [12, 100],
            "power_range": [100, 5000],
        },
        "push_pull": {
            "vin_range": [12, 100],
            "vout_range": [5, 48],
            "power_range": [20, 500],
        },
    }
    
    # Generate examples for each topology
    examples_per_topology = num_examples // len(topologies)
    
    for topo, params in topologies.items():
        for i in range(examples_per_topology):
            example = generate_single_example(topo, params, i)
            examples.append(example)
    
    random.shuffle(examples)
    return examples


def generate_single_example(topology: str, params: Dict, idx: int) -> Dict:
    """Generate a single training example with full solution."""
    # Random parameters within range
    if topology == "buck":
        vin = random.uniform(*params["vin_range"])
        vout = vin * random.uniform(*params["vout_ratio_range"])
        power = random.uniform(*params["power_range"])
        iout = power / vout
        D = vout / vin
        formula = f"D = Vout/Vin = {vout:.1f}/{vin:.1f} = {D:.4f}"
        
    elif topology == "boost":
        vin = random.uniform(*params["vin_range"])
        vout = vin * random.uniform(*params["vout_ratio_range"])
        power = random.uniform(*params["power_range"])
        iout = power / vout
        D = 1 - vin / vout
        formula = f"D = 1 - Vin/Vout = 1 - {vin:.1f}/{vout:.1f} = {D:.4f}"
        
    elif topology == "buck_boost":
        vin = random.uniform(*params["vin_range"])
        vout = random.uniform(*params["vout_range"])
        power = random.uniform(*params["power_range"])
        iout = power / abs(vout)
        D = abs(vout) / (vin + abs(vout))
        formula = f"D = |Vout|/(Vin + |Vout|) = {abs(vout):.1f}/({vin:.1f} + {abs(vout):.1f}) = {D:.4f}"
        
    elif topology == "sepic":
        vin = random.uniform(*params["vin_range"])
        vout = random.uniform(*params["vout_range"])
        power = random.uniform(*params["power_range"])
        iout = power / vout
        D = vout / (vin + vout)
        formula = f"D = Vout/(Vin + Vout) = {vout:.1f}/({vin:.1f} + {vout:.1f}) = {D:.4f}"
        
    elif topology == "cuk":
        vin = random.uniform(*params["vin_range"])
        vout = random.uniform(*params["vout_range"])
        power = random.uniform(*params["power_range"])
        iout = power / abs(vout)
        D = abs(vout) / (vin + abs(vout))
        formula = f"D = |Vout|/(Vin + |Vout|) = {abs(vout):.1f}/({vin:.1f} + {abs(vout):.1f}) = {D:.4f}"
        
    elif topology in ["flyback", "forward", "half_bridge", "full_bridge", "push_pull"]:
        vin = random.uniform(*params["vin_range"])
        vout = random.uniform(*params["vout_range"])
        power = random.uniform(*params["power_range"])
        iout = power / vout
        n = round(random.uniform(0.5, 3.0), 1)  # turns ratio
        
        if topology == "flyback":
            D = vout / (vout + vin * n)
            formula = f"D = Vout/(Vout + Vin×n) = {vout:.1f}/({vout:.1f} + {vin:.1f}×{n}) = {D:.4f}"
        else:
            D = vout / (vin * n)
            D = min(0.9, max(0.1, D))
            formula = f"D = Vout/(Vin×n) = {vout:.1f}/({vin:.1f}×{n}) = {D:.4f}"
    else:
        vin, vout, power, iout, D, n = 12, 5, 10, 2, 0.417, 1
        formula = "D = Vout/Vin"
    
    # Ensure D is in valid range
    D = max(0.05, min(0.95, D))
    
    # Common parameters
    fsw = random.choice([50e3, 100e3, 200e3, 250e3, 500e3])
    R_load = vout / iout if iout > 0 else 10
    
    # Calculate inductor and capacitor
    ripple_pct = random.uniform(3, 10)
    L = (vin * D) / (2 * fsw * iout * 0.3) if iout > 0 else 100e-6
    L = max(L, 1e-6)  # minimum 1µH
    C = iout * D / (fsw * vout * ripple_pct / 100) if vout > 0 else 100e-6
    C = max(C, 1e-6)  # minimum 1µF
    
    # Create prompt
    prompt = f"Design a {topology.replace('_', ' ')} converter: {vin:.1f}V input to {vout:.1f}V output, {iout:.1f}A load current"
    
    # Create detailed solution
    solution = f"""**Problem Analysis:**
- Input Voltage: Vin = {vin:.1f}V
- Output Voltage: Vout = {vout:.1f}V  
- Output Current: Iout = {iout:.2f}A
- Output Power: P = {power:.1f}W
- Switching Frequency: fsw = {fsw/1000:.0f}kHz

**Topology:** {topology.upper()} converter

**Step 1: Duty Cycle Calculation**
{formula}

Duty Cycle: D = {D:.4f} ({D*100:.1f}%)

**Step 2: Inductor Selection (CCM, 30% ripple)**
L = Vin × D / (2 × fsw × ΔIL)
L = {vin:.1f} × {D:.4f} / (2 × {fsw/1000:.0f}k × {iout*0.3:.2f})
L = {L*1e6:.1f}µH

**Step 3: Output Capacitor (for {ripple_pct:.0f}% ripple)**
C = Iout × D / (fsw × ΔVout)
C = {iout:.2f} × {D:.4f} / ({fsw/1000:.0f}k × {vout*ripple_pct/100:.3f})
C = {C*1e6:.1f}µF

**Step 4: Load Resistance**
R_load = Vout / Iout = {vout:.1f} / {iout:.2f} = {R_load:.2f}Ω

**Final Design:**
- Duty Cycle: D = {D:.4f}
- Inductance: L = {L*1e6:.1f}µH  
- Capacitance: C = {C*1e6:.1f}µF
- Load: R = {R_load:.2f}Ω
- Expected Vout: {vout:.1f}V"""

    system_prompt = """You are an expert power electronics engineer. When designing DC-DC converters:

1. ALWAYS calculate duty cycle first using the correct formula:
   - Buck: D = Vout/Vin
   - Boost: D = 1 - Vin/Vout
   - Buck-Boost: D = |Vout|/(Vin + |Vout|)
   - SEPIC: D = Vout/(Vin + Vout)
   - Flyback: D = Vout/(Vout + Vin×n)

2. Show all calculations step-by-step
3. Size inductor for CCM operation
4. Size capacitor for acceptable ripple"""

    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": solution}
        ]
    }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 70)
    print("POWERELECLLM BENCHMARK ENHANCEMENT")
    print("=" * 70)
    
    # 1. Generate Level 5 expert problems
    print("\n[1/3] Generating Level 5 Expert Problems...")
    level5_problems = generate_level5_problems()
    
    level5_dir = PROJECT_ROOT / "benchmarks" / "level_5"
    level5_dir.mkdir(exist_ok=True)
    
    with open(level5_dir / "problems_expert.json", "w") as f:
        json.dump({
            "level": 5,
            "description": "Expert-level problems: resonant converters, multi-phase, cascaded, soft-switching",
            "problems": level5_problems
        }, f, indent=2)
    
    print(f"   ✓ Created {len(level5_problems)} Level 5 expert problems")
    
    # 2. Generate harder Level 4 problems  
    print("\n[2/3] Generating Harder Level 4+ Problems...")
    harder_problems = generate_harder_l4_problems()
    
    with open(PROJECT_ROOT / "benchmarks" / "level_4" / "problems_advanced.json", "w") as f:
        json.dump({
            "level": 4,
            "description": "Advanced multi-constraint optimization problems",
            "problems": harder_problems
        }, f, indent=2)
    
    print(f"   ✓ Created {len(harder_problems)} advanced Level 4 problems")
    
    # 3. Generate large fine-tuning dataset
    print("\n[3/3] Generating Large Fine-tuning Dataset (2000 examples)...")
    finetune_data = generate_large_finetune_dataset(2000)
    
    finetune_dir = PROJECT_ROOT / "benchmarks" / "finetune"
    finetune_dir.mkdir(exist_ok=True)
    
    # Split 90/10
    split_idx = int(len(finetune_data) * 0.9)
    train_data = finetune_data[:split_idx]
    val_data = finetune_data[split_idx:]
    
    with open(finetune_dir / "train_large.jsonl", "w") as f:
        for ex in train_data:
            f.write(json.dumps(ex) + "\n")
    
    with open(finetune_dir / "val_large.jsonl", "w") as f:
        for ex in val_data:
            f.write(json.dumps(ex) + "\n")
    
    print(f"   ✓ Created {len(train_data)} training examples")
    print(f"   ✓ Created {len(val_data)} validation examples")
    
    # Summary
    print("\n" + "=" * 70)
    print("ENHANCEMENT COMPLETE")
    print("=" * 70)
    print(f"""
New Files Created:
  - benchmarks/level_5/problems_expert.json ({len(level5_problems)} problems)
  - benchmarks/level_4/problems_advanced.json ({len(harder_problems)} problems)
  - benchmarks/finetune/train_large.jsonl ({len(train_data)} examples)
  - benchmarks/finetune/val_large.jsonl ({len(val_data)} examples)

Total New Problems: {len(level5_problems) + len(harder_problems)}
Total Fine-tuning Examples: {len(finetune_data)}
""")


if __name__ == "__main__":
    main()
