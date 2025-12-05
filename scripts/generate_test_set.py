#!/usr/bin/env python3
"""
Generate 150 NEW test problems for PowerElecBench
These problems are HELD OUT from training - never fine-tune on these!

Distribution:
- Level 1: 45 problems (basic)
- Level 2: 45 problems (intermediate)  
- Level 3: 35 problems (advanced)
- Level 4: 25 problems (expert)
"""

import json
import random
import math
from pathlib import Path
from datetime import datetime

# Set seed for reproducibility but different from training data
random.seed(42424242)

BENCHMARK_DIR = Path(__file__).parent.parent / "benchmarks" / "test_set"

# Topology equations and configs
TOPOLOGIES = {
    "buck": {
        "vout_eq": lambda vin, d: vin * d,
        "d_range": (0.1, 0.9),
        "vin_range": (12, 48),
        "description": "step-down DC-DC converter"
    },
    "boost": {
        "vout_eq": lambda vin, d: vin / (1 - d),
        "d_range": (0.1, 0.8),
        "vin_range": (5, 24),
        "description": "step-up DC-DC converter"
    },
    "buck_boost": {
        "vout_eq": lambda vin, d: -vin * d / (1 - d),  # Inverted output
        "d_range": (0.2, 0.7),
        "vin_range": (12, 36),
        "description": "inverting DC-DC converter"
    },
    "flyback": {
        "vout_eq": lambda vin, d, n=1: vin * d * n / (1 - d),
        "d_range": (0.2, 0.6),
        "vin_range": (100, 400),  # AC rectified
        "description": "isolated flyback converter"
    },
    "forward": {
        "vout_eq": lambda vin, d, n=0.5: vin * d * n,
        "d_range": (0.2, 0.45),
        "vin_range": (100, 400),
        "description": "isolated forward converter"
    },
    "half_bridge": {
        "vout_eq": lambda vin, d, n=0.5: vin * d * n,
        "d_range": (0.3, 0.5),
        "vin_range": (200, 400),
        "description": "half-bridge DC-DC converter"
    },
    "full_bridge": {
        "vout_eq": lambda vin, d, n=0.25: vin * 2 * d * n,
        "d_range": (0.3, 0.5),
        "vin_range": (300, 600),
        "description": "full-bridge DC-DC converter"
    }
}

# Standard voltage rails
STANDARD_VOUT = [1.2, 1.5, 1.8, 2.5, 3.3, 5.0, 9.0, 12.0, 15.0, 18.0, 24.0, 28.0, 36.0, 48.0]

# Power levels
POWER_LEVELS = {
    1: (1, 50),      # 1-50W
    2: (10, 200),    # 10-200W
    3: (50, 500),    # 50-500W
    4: (100, 2000)   # 100-2000W
}

# Switching frequencies
FSW_RANGES = {
    1: (50e3, 200e3),
    2: (100e3, 500e3),
    3: (50e3, 300e3),
    4: (20e3, 150e3)
}

def calculate_components(topology, vin, vout, power, fsw, level):
    """Calculate component values based on topology and specs"""
    
    iout = power / abs(vout)
    
    # Ripple targets (tighter for higher levels)
    ripple_i_pct = {1: 0.4, 2: 0.3, 3: 0.2, 4: 0.15}[level]
    ripple_v_pct = {1: 0.02, 2: 0.015, 3: 0.01, 4: 0.005}[level]
    
    delta_il = iout * ripple_i_pct
    delta_vout = abs(vout) * ripple_v_pct
    
    if topology == "buck":
        d = vout / vin
        L = (vin - vout) * d / (fsw * delta_il)
        C = delta_il / (8 * fsw * delta_vout)
    elif topology == "boost":
        d = 1 - vin / vout
        L = vin * d / (fsw * delta_il)
        C = iout * d / (fsw * delta_vout)
    elif topology == "buck_boost":
        d = abs(vout) / (vin + abs(vout))
        L = vin * d / (fsw * delta_il)
        C = iout * d / (fsw * delta_vout)
    elif topology in ["flyback", "forward", "half_bridge", "full_bridge"]:
        # Simplified for isolated topologies
        d = 0.4
        L = vin * d / (fsw * delta_il) * 0.5
        C = iout * 0.5 / (fsw * delta_vout)
    else:
        d = 0.5
        L = 100e-6
        C = 100e-6
    
    # Clamp to reasonable values
    L = max(1e-6, min(10e-3, L))
    C = max(1e-6, min(10e-3, C))
    d = max(0.1, min(0.9, d))
    
    return {
        "L": round(L, 9),
        "C": round(C, 9),
        "D": round(d, 4),
        "R_load": round(vout**2 / power, 3),
        "fsw": int(fsw)
    }

def generate_level1_problem(idx):
    """Basic: Single topology, clear specs, standard values"""
    topology = random.choice(["buck", "boost"])
    config = TOPOLOGIES[topology]
    
    if topology == "buck":
        vin = random.choice([24, 36, 48])
        vout = random.choice([v for v in STANDARD_VOUT if v < vin * 0.9])
    else:
        vin = random.choice([12, 18, 24])
        vout_options = [v for v in STANDARD_VOUT if v > vin * 1.2 and v < vin * 4]
        if not vout_options:
            vout_options = [vin * 2]  # Fallback
        vout = random.choice(vout_options)
    
    power = random.choice([5, 10, 15, 20, 25, 30, 50])
    fsw = random.choice([100e3, 150e3, 200e3])
    
    components = calculate_components(topology, vin, vout, power, fsw, 1)
    
    prompt = f"""Design a {topology.replace('_', ' ')} converter with the following specifications:
- Input voltage: {vin}V DC
- Output voltage: {vout}V DC
- Output power: {power}W
- Switching frequency: {int(fsw/1000)}kHz

Calculate the required component values (inductor, capacitor, duty cycle) and verify the output voltage."""

    return {
        "id": f"TEST_L1_{idx:03d}",
        "level": 1,
        "topology": topology,
        "prompt": prompt,
        "specs": {
            "vin": vin,
            "vout": vout,
            "power": power,
            "fsw": fsw
        },
        "solution": {
            "topology": topology,
            "vout": vout,
            "components": components,
            "equations": f"Vout = Vin × D = {vin} × {components['D']:.3f} = {vout}V" if topology == "buck" else f"Vout = Vin / (1-D) = {vin} / (1-{components['D']:.3f}) = {vout}V"
        }
    }

def generate_level2_problem(idx):
    """Intermediate: More topologies, efficiency constraints"""
    topology = random.choice(["buck", "boost", "buck_boost", "flyback"])
    config = TOPOLOGIES[topology]
    
    vin = random.randint(config["vin_range"][0], config["vin_range"][1])
    
    if topology == "buck":
        vout = random.choice([v for v in STANDARD_VOUT if v < vin * 0.85])
    elif topology == "boost":
        vout = random.choice([v for v in STANDARD_VOUT if v > vin * 1.3 and v < vin * 4])
    elif topology == "buck_boost":
        vout = -random.choice([v for v in STANDARD_VOUT if v < 30])
    else:  # flyback
        vout = random.choice([5, 12, 15, 24])
    
    power = random.randint(POWER_LEVELS[2][0], POWER_LEVELS[2][1])
    fsw = random.randint(int(FSW_RANGES[2][0]), int(FSW_RANGES[2][1]))
    efficiency_target = random.choice([90, 92, 94, 95])
    
    components = calculate_components(topology, vin, abs(vout), power, fsw, 2)
    
    ripple_req = random.choice(["1%", "2%", "50mV", "100mV"])
    
    prompt = f"""Design a {config['description']} with these requirements:
- Input voltage: {vin}V DC
- Output voltage: {abs(vout)}V DC{' (inverted)' if vout < 0 else ''}
- Output power: {power}W
- Switching frequency: {fsw/1000:.0f}kHz
- Target efficiency: >{efficiency_target}%
- Output voltage ripple: <{ripple_req}

Provide complete component values and verify your design meets all specifications."""

    return {
        "id": f"TEST_L2_{idx:03d}",
        "level": 2,
        "topology": topology,
        "prompt": prompt,
        "specs": {
            "vin": vin,
            "vout": abs(vout),
            "power": power,
            "fsw": fsw,
            "efficiency": efficiency_target
        },
        "solution": {
            "topology": topology,
            "vout": abs(vout),
            "components": components
        }
    }

def generate_level3_problem(idx):
    """Advanced: More complex specs, efficiency constraints, ripple requirements"""
    
    # Use topologies we can simulate accurately
    topology = random.choice(["buck", "boost", "buck_boost"])
    
    scenario = random.choice([
        "solar_mppt",
        "battery_charger",
        "led_driver",
        "industrial_supply",
        "telecom_power"
    ])
    
    if scenario == "solar_mppt" and topology == "boost":
        vin = random.randint(25, 40)  # Panel voltage
        vout = random.choice([48, 56, 60])
        power = random.randint(100, 300)
        prompt_context = "solar MPPT charge controller"
        extra_req = f"Input voltage range: {vin-10}V to {vin+10}V"
    elif scenario == "battery_charger":
        if topology == "buck":
            vin = random.choice([24, 36, 48])
            vout = random.choice([12, 14.4, 24, 28.8])  # Battery charging voltages
            power = random.randint(50, 200)
            prompt_context = "battery charger"
            extra_req = "CC/CV charging capability"
        else:
            vin = random.choice([12, 24])
            vout = random.choice([36, 48])
            power = random.randint(50, 150)
            prompt_context = "battery charger (boost stage)"
            extra_req = "Over-voltage protection required"
    elif scenario == "led_driver":
        if topology == "buck":
            vin = random.choice([24, 36, 48])
            vout = random.randint(12, 36)  # LED string voltage
            power = random.randint(20, 100)
            prompt_context = "constant-current LED driver"
            extra_req = f"Output current: {power/vout:.2f}A ±2%"
        else:
            vin = random.choice([12, 24])
            vout = random.choice([36, 48])
            power = random.randint(30, 100)
            prompt_context = "LED driver (boost stage)"
            extra_req = "PWM dimming support"
    elif scenario == "industrial_supply":
        if topology == "buck":
            vin = random.choice([24, 36, 48])
            vout = random.choice([5, 12, 15, 24])
            power = random.randint(50, 200)
            prompt_context = "industrial DC power supply"
            extra_req = "Wide temperature range: -40°C to +85°C"
        else:
            vin = random.choice([12, 24])
            vout = random.choice([36, 48])
            power = random.randint(50, 200)
            prompt_context = "industrial boost supply"
            extra_req = "Input undervoltage lockout"
    else:  # telecom_power
        if topology == "buck":
            vin = random.choice([48, 54])
            vout = random.choice([12, 24])
            power = random.randint(100, 300)
            prompt_context = "telecom power module"
            extra_req = "Hot-swap capability"
        else:
            vin = random.choice([24, 36])
            vout = random.choice([48, 54])
            power = random.randint(100, 250)
            prompt_context = "telecom boost module"
            extra_req = "N+1 redundancy support"
    
    fsw = random.choice([100e3, 150e3, 200e3])
    
    # Calculate correct duty cycle for solution
    if topology == "buck":
        duty = vout / vin
    elif topology == "boost":
        duty = 1 - vin / vout
    else:  # buck_boost
        duty = abs(vout) / (vin + abs(vout))
    
    components = calculate_components(topology, vin, abs(vout), power, fsw, 3)
    components["D"] = round(duty, 4)
    
    ripple_req = random.choice(["1%", "2%", "50mV", "100mV"])
    efficiency_target = random.choice([90, 92, 94])
    
    prompt = f"""Design a {prompt_context} ({topology.replace('_', '-')} topology) with these specifications:
- Input voltage: {vin}V DC
- Output voltage: {abs(vout)}V DC{' (magnitude)' if topology == 'buck_boost' else ''}
- Output power: {power}W
- Switching frequency: {fsw/1000:.0f}kHz
- {extra_req}
- Target efficiency: >{efficiency_target}%
- Output voltage ripple: <{ripple_req}

Provide:
1. Duty cycle calculation
2. Inductor value (L) with units
3. Output capacitor value (C) with units
4. Verify output voltage calculation"""

    return {
        "id": f"TEST_L3_{idx:03d}",
        "level": 3,
        "topology": topology,
        "prompt": prompt,
        "specs": {
            "vin": vin,
            "vout": abs(vout),
            "power": power,
            "fsw": fsw,
            "efficiency": efficiency_target,
            "scenario": scenario
        },
        "solution": {
            "topology": topology,
            "vout": abs(vout),
            "components": components
        }
    }

def generate_level4_problem(idx):
    """Expert: Complete system design with multi-criteria requirements"""
    
    # Use topologies we can simulate - but make problems harder
    topology = random.choice(["buck", "boost", "buck_boost"])
    
    scenario = random.choice([
        "data_center",
        "aerospace",
        "medical",
        "automotive"
    ])
    
    if scenario == "data_center":
        if topology == "buck":
            vin = random.choice([48, 54])
            vout = random.choice([3.3, 5, 12])
            power = random.randint(200, 500)
        else:
            vin = random.choice([24, 36])
            vout = random.choice([48, 54])
            power = random.randint(200, 400)
        prompt = f"""Design a {power}W data center power module ({topology.replace('_', '-')} topology):

REQUIREMENTS:
- Input: {vin}V DC ±5%
- Output: {vout}V DC
- Efficiency: >95% at 50% load, >93% at full load
- Output voltage ripple: <1% peak-to-peak
- Switching frequency: 200-300kHz
- Transient response: <3% deviation for 25% load step

DELIVERABLES:
1. Duty cycle calculation with equation
2. Inductor value (L) sized for <30% current ripple
3. Output capacitor (C) sized for ripple specification
4. Load resistance for given power level
5. Verify steady-state output voltage"""
        
    elif scenario == "aerospace":
        if topology == "buck":
            vin = 28  # MIL-STD-704
            vout = random.choice([3.3, 5, 12, 15])
            power = random.randint(20, 100)
        else:
            vin = random.choice([12, 18])
            vout = 28
            power = random.randint(20, 80)
        prompt = f"""Design a MIL-STD-704 compliant {topology.replace('_', '-')} converter for avionics:

REQUIREMENTS:
- Input: {vin}V DC nominal (per MIL-STD-704)
- Output: {vout}V at {power}W
- Operating temp: -55°C to +85°C
- Output voltage tolerance: ±2%
- Ripple: <0.5% peak-to-peak
- Switching frequency: 100-200kHz

DELIVERABLES:
1. Complete duty cycle analysis
2. Inductor selection with ripple calculation
3. Capacitor selection with ESR consideration
4. Worst-case output voltage verification"""

    elif scenario == "medical":
        if topology == "buck":
            vin = random.choice([24, 36])
            vout = random.choice([5, 12, 15])
            power = random.randint(15, 60)
        else:
            vin = random.choice([12, 15])
            vout = random.choice([24, 36])
            power = random.randint(20, 50)
        prompt = f"""Design a medical-grade {topology.replace('_', '-')} power supply:

REQUIREMENTS:
- Input: {vin}V DC ±10%
- Output: {vout}V DC at {power}W
- Output ripple: <50mV peak-to-peak
- Load regulation: <1%
- Switching frequency: 150-250kHz
- Low EMI (consider input filtering)

DELIVERABLES:
1. Duty cycle calculation for nominal and worst-case input
2. Inductor value for continuous conduction mode
3. Output capacitor for ripple specification
4. Verify output at minimum and maximum input voltage"""

    else:  # automotive
        if topology == "buck":
            vin = random.choice([12, 24, 48])
            vout = random.choice([3.3, 5, 12])
            power = random.randint(30, 150)
        else:
            vin = 12
            vout = random.choice([24, 36, 48])
            power = random.randint(50, 200)
        prompt = f"""Design an automotive {topology.replace('_', '-')} converter (AEC-Q100 grade):

REQUIREMENTS:
- Input: {vin}V nominal ({int(vin*0.7)}V to {int(vin*1.5)}V range)
- Output: {vout}V at {power}W
- Operating temp: -40°C to +125°C
- Output ripple: <2% peak-to-peak
- Switching frequency: 200-400kHz
- Load dump transient immunity

DELIVERABLES:
1. Duty cycle at nominal, min, and max input
2. Inductor value for CCM operation across input range
3. Output capacitor for ripple and transient response
4. Verify operation at voltage extremes"""

    fsw = random.choice([150e3, 200e3, 250e3, 300e3])
    
    # Calculate correct duty cycle
    if topology == "buck":
        duty = vout / vin
    elif topology == "boost":
        duty = 1 - vin / vout
    else:  # buck_boost
        duty = abs(vout) / (vin + abs(vout))
    
    components = calculate_components(topology, vin, abs(vout), power, fsw, 4)
    components["D"] = round(duty, 4)
    
    return {
        "id": f"TEST_L4_{idx:03d}",
        "level": 4,
        "topology": topology,
        "prompt": prompt,
        "specs": {
            "vin": vin,
            "vout": abs(vout),
            "power": power,
            "fsw": fsw,
            "scenario": scenario
        },
        "solution": {
            "topology": topology,
            "vout": abs(vout),
            "components": components
        }
    }

def main():
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    
    all_problems = []
    
    # Generate Level 1: 45 problems
    print("Generating Level 1 test problems (45)...")
    l1_problems = [generate_level1_problem(i+1) for i in range(45)]
    all_problems.extend(l1_problems)
    
    # Generate Level 2: 45 problems  
    print("Generating Level 2 test problems (45)...")
    l2_problems = [generate_level2_problem(i+1) for i in range(45)]
    all_problems.extend(l2_problems)
    
    # Generate Level 3: 35 problems
    print("Generating Level 3 test problems (35)...")
    l3_problems = [generate_level3_problem(i+1) for i in range(35)]
    all_problems.extend(l3_problems)
    
    # Generate Level 4: 25 problems
    print("Generating Level 4 test problems (25)...")
    l4_problems = [generate_level4_problem(i+1) for i in range(25)]
    all_problems.extend(l4_problems)
    
    # Save all problems
    output_file = BENCHMARK_DIR / "test_problems.json"
    with open(output_file, "w") as f:
        json.dump({
            "metadata": {
                "name": "PowerElecBench Test Set",
                "version": "1.0",
                "description": "150 held-out test problems - DO NOT TRAIN ON THESE",
                "created": datetime.now().isoformat(),
                "distribution": {
                    "level_1": 45,
                    "level_2": 45,
                    "level_3": 35,
                    "level_4": 25
                }
            },
            "problems": all_problems
        }, f, indent=2)
    
    print(f"\n✅ Generated {len(all_problems)} test problems")
    print(f"📁 Saved to: {output_file}")
    
    # Summary
    print("\n=== Test Set Summary ===")
    for level in [1, 2, 3, 4]:
        count = len([p for p in all_problems if p["level"] == level])
        print(f"Level {level}: {count} problems")
    print(f"TOTAL: {len(all_problems)} problems")
    print("\n⚠️  IMPORTANT: Do NOT fine-tune on these problems!")

if __name__ == "__main__":
    main()
