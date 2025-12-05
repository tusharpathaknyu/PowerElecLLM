#!/usr/bin/env python3
"""
Generate 150 NEW test problems for PowerElecBench v2
WITH ALL TOPOLOGIES - not just buck/boost/buck_boost

Distribution:
- Level 1: 45 problems (basic) - buck, boost
- Level 2: 45 problems (intermediate) - buck, boost, buck_boost, flyback, forward
- Level 3: 35 problems (advanced) - ALL topologies
- Level 4: 25 problems (expert) - ALL topologies with complex constraints
"""

import json
import random
import math
from pathlib import Path
from datetime import datetime

# Different seed from v1
random.seed(98765432)

BENCHMARK_DIR = Path(__file__).parent.parent / "benchmarks" / "test_set_v2"

# ALL TOPOLOGIES with proper equations
TOPOLOGIES = {
    "buck": {
        "vout_eq": lambda vin, d, n=1: vin * d,
        "d_range": (0.1, 0.9),
        "vin_range": (12, 48),
        "description": "step-down DC-DC converter",
        "levels": [1, 2, 3, 4]
    },
    "boost": {
        "vout_eq": lambda vin, d, n=1: vin / (1 - d),
        "d_range": (0.1, 0.8),
        "vin_range": (5, 24),
        "description": "step-up DC-DC converter",
        "levels": [1, 2, 3, 4]
    },
    "buck_boost": {
        "vout_eq": lambda vin, d, n=1: -vin * d / (1 - d),
        "d_range": (0.2, 0.7),
        "vin_range": (12, 36),
        "description": "inverting DC-DC converter",
        "levels": [2, 3, 4]
    },
    "flyback": {
        "vout_eq": lambda vin, d, n=1: vin * d * n / (1 - d),
        "d_range": (0.2, 0.5),
        "vin_range": (100, 400),
        "description": "isolated flyback converter",
        "n_range": (0.1, 0.5),
        "levels": [2, 3, 4]
    },
    "forward": {
        "vout_eq": lambda vin, d, n=0.5: vin * d * n,
        "d_range": (0.2, 0.45),
        "vin_range": (100, 400),
        "description": "isolated forward converter",
        "n_range": (0.3, 0.8),
        "levels": [2, 3, 4]
    },
    "sepic": {
        "vout_eq": lambda vin, d, n=1: vin * d / (1 - d),
        "d_range": (0.2, 0.7),
        "vin_range": (12, 48),
        "description": "SEPIC converter (non-inverting buck-boost)",
        "levels": [3, 4]
    },
    "cuk": {
        "vout_eq": lambda vin, d, n=1: -vin * d / (1 - d),
        "d_range": (0.2, 0.7),
        "vin_range": (12, 48),
        "description": "Ćuk converter (inverting with continuous current)",
        "levels": [3, 4]
    },
    "half_bridge": {
        "vout_eq": lambda vin, d, n=0.5: vin * d * n,
        "d_range": (0.35, 0.5),
        "vin_range": (200, 400),
        "description": "half-bridge DC-DC converter",
        "n_range": (0.2, 0.6),
        "levels": [3, 4]
    },
    "full_bridge": {
        "vout_eq": lambda vin, d, n=0.25: vin * 2 * d * n,
        "d_range": (0.35, 0.5),
        "vin_range": (300, 600),
        "description": "full-bridge DC-DC converter",
        "n_range": (0.1, 0.4),
        "levels": [3, 4]
    },
    "push_pull": {
        "vout_eq": lambda vin, d, n=0.5: 2 * vin * d * n,
        "d_range": (0.3, 0.5),
        "vin_range": (24, 100),
        "description": "push-pull DC-DC converter",
        "n_range": (0.2, 0.6),
        "levels": [4]
    }
}

STANDARD_VOUT = [1.2, 1.5, 1.8, 2.5, 3.3, 5.0, 9.0, 12.0, 15.0, 18.0, 24.0, 28.0, 36.0, 48.0]

POWER_LEVELS = {
    1: (5, 50),
    2: (20, 200),
    3: (50, 500),
    4: (100, 2000)
}

FSW_RANGES = {
    1: (50e3, 200e3),
    2: (100e3, 500e3),
    3: (50e3, 300e3),
    4: (20e3, 150e3)
}


def get_topologies_for_level(level):
    """Get available topologies for a level"""
    return [t for t, cfg in TOPOLOGIES.items() if level in cfg.get("levels", [1,2,3,4])]


def calculate_duty_cycle(topology, vin, vout, n=1):
    """Calculate duty cycle for a topology"""
    vout_abs = abs(vout)
    
    if topology == "buck":
        return vout_abs / vin
    elif topology == "boost":
        return 1 - vin / vout_abs
    elif topology in ["buck_boost", "cuk"]:
        return vout_abs / (vin + vout_abs)
    elif topology == "sepic":
        return vout_abs / (vin + vout_abs)
    elif topology == "flyback":
        return vout_abs / (vout_abs + vin * n)
    elif topology == "forward":
        return vout_abs / (vin * n) if vin * n > 0 else 0.4
    elif topology == "half_bridge":
        return vout_abs / (vin * n) if vin * n > 0 else 0.4
    elif topology == "full_bridge":
        return vout_abs / (2 * vin * n) if vin * n > 0 else 0.4
    elif topology == "push_pull":
        return vout_abs / (2 * vin * n) if vin * n > 0 else 0.4
    return 0.5


def calculate_components(topology, vin, vout, power, fsw, level, n=1):
    """Calculate component values"""
    vout_abs = abs(vout)
    iout = power / vout_abs
    
    ripple_i_pct = {1: 0.4, 2: 0.3, 3: 0.2, 4: 0.15}[level]
    ripple_v_pct = {1: 0.02, 2: 0.015, 3: 0.01, 4: 0.005}[level]
    
    delta_il = iout * ripple_i_pct
    delta_vout = vout_abs * ripple_v_pct
    
    d = calculate_duty_cycle(topology, vin, vout, n)
    d = max(0.1, min(0.9, d))
    
    # Calculate L and C based on topology
    if topology == "buck":
        L = (vin - vout_abs) * d / (fsw * delta_il) if delta_il > 0 else 100e-6
        C = delta_il / (8 * fsw * delta_vout) if delta_vout > 0 else 100e-6
    elif topology == "boost":
        L = vin * d / (fsw * delta_il) if delta_il > 0 else 100e-6
        C = iout * d / (fsw * delta_vout) if delta_vout > 0 else 100e-6
    elif topology in ["buck_boost", "cuk", "sepic"]:
        L = vin * d / (fsw * delta_il) if delta_il > 0 else 100e-6
        C = iout * d / (fsw * delta_vout) if delta_vout > 0 else 100e-6
    else:  # Isolated topologies
        L = vin * d / (fsw * delta_il * 2) if delta_il > 0 else 100e-6
        C = iout * 0.5 / (fsw * delta_vout) if delta_vout > 0 else 100e-6
    
    L = max(1e-6, min(10e-3, L))
    C = max(1e-6, min(10e-3, C))
    
    return {
        "L": round(L, 9),
        "C": round(C, 9),
        "D": round(d, 4),
        "R_load": round(vout_abs**2 / power, 3),
        "fsw": int(fsw),
        "n": n if topology in ["flyback", "forward", "half_bridge", "full_bridge", "push_pull"] else None
    }


def generate_problem(level, idx):
    """Generate a problem for a given level"""
    topologies = get_topologies_for_level(level)
    topology = random.choice(topologies)
    config = TOPOLOGIES[topology]
    
    # Get vin
    vin = random.randint(config["vin_range"][0], config["vin_range"][1])
    
    # Get turns ratio for isolated
    n = 1
    if "n_range" in config:
        n = round(random.uniform(config["n_range"][0], config["n_range"][1]), 2)
    
    # Calculate achievable vout based on topology and d range
    d_min, d_max = config["d_range"]
    d = round(random.uniform(d_min, d_max), 3)
    
    # Calculate vout from equation
    try:
        if topology in ["flyback", "forward", "half_bridge", "full_bridge", "push_pull"]:
            vout_calc = config["vout_eq"](vin, d, n)
        else:
            vout_calc = config["vout_eq"](vin, d)
    except:
        vout_calc = 12
    
    # Round to nearest standard value or use calculated
    vout_calc = abs(vout_calc)
    closest = min(STANDARD_VOUT, key=lambda x: abs(x - vout_calc))
    
    # Use closest standard voltage if within 20%, else use calculated
    if abs(closest - vout_calc) / vout_calc < 0.2:
        vout = closest
    else:
        vout = round(vout_calc, 1)
    
    # For inverting topologies
    if topology in ["buck_boost", "cuk"]:
        vout = -vout
    
    # Power
    power = random.randint(POWER_LEVELS[level][0], POWER_LEVELS[level][1])
    
    # Frequency
    fsw = random.randint(int(FSW_RANGES[level][0]), int(FSW_RANGES[level][1]))
    
    # Calculate components
    components = calculate_components(topology, vin, vout, power, fsw, level, n)
    
    # Generate prompt based on level
    if level == 1:
        prompt = f"""Design a {topology.replace('_', ' ')} converter with:
- Input voltage: {vin}V DC
- Output voltage: {abs(vout)}V DC
- Output power: {power}W
- Switching frequency: {int(fsw/1000)}kHz

Calculate the duty cycle, inductor value, and capacitor value."""

    elif level == 2:
        efficiency_target = random.choice([90, 92, 95])
        prompt = f"""Design a {topology.replace('_', ' ')} converter meeting these specifications:
- Input voltage: {vin}V DC
- Output voltage: {abs(vout)}V DC{'(inverted)' if vout < 0 else ''}
- Output power: {power}W
- Switching frequency: {int(fsw/1000)}kHz
- Target efficiency: >{efficiency_target}%

Provide duty cycle, L, C values and verify Vout."""

    elif level == 3:
        ripple_target = random.choice([1, 2, 3, 5])
        current_ripple = random.choice([20, 30, 40])
        prompt = f"""Design a {topology.replace('_', ' ')} converter with these requirements:
- Input: {vin}V DC
- Output: {abs(vout)}V DC{' (inverted polarity)' if vout < 0 else ''}
- Power: {power}W
- Frequency: {int(fsw/1000)}kHz
- Output voltage ripple: <{ripple_target}%
- Inductor current ripple: <{current_ripple}% of average

Calculate all component values ensuring CCM operation."""
        if n != 1:
            prompt += f"\n- Transformer turns ratio: {n}"

    else:  # Level 4
        ripple_target = random.choice([0.5, 1, 2])
        current_ripple = random.choice([15, 20, 25])
        vin_min = int(vin * 0.9)
        vin_max = int(vin * 1.1)
        prompt = f"""Design a high-performance {topology.replace('_', ' ')} converter:

Specifications:
- Input voltage range: {vin_min}V to {vin_max}V DC
- Output voltage: {abs(vout)}V DC{' (negative rail)' if vout < 0 else ''} ±1%
- Output power: {power}W
- Switching frequency: {int(fsw/1000)}kHz
- Output ripple: <{ripple_target}% peak-to-peak
- Current ripple: <{current_ripple}% for CCM
- Efficiency target: >92%

Provide:
1. Duty cycle range for input variation
2. Minimum inductance for CCM at light load (20% rated)
3. Output capacitor for ripple spec
4. Component stress analysis"""
        if n != 1:
            prompt += f"\n- Transformer turns ratio: {n}"
    
    return {
        "id": f"TEST_L{level}_{idx:03d}",
        "level": level,
        "topology": topology,
        "prompt": prompt,
        "specs": {
            "vin": vin,
            "vout": vout,
            "power": power,
            "fsw": fsw,
            "n": n if n != 1 else None
        },
        "solution": {
            "topology": topology,
            "vout": vout,
            "components": components
        }
    }


def main():
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    
    problems = []
    
    # Level distribution
    distribution = {1: 45, 2: 45, 3: 35, 4: 25}
    
    for level, count in distribution.items():
        print(f"Generating Level {level}: {count} problems")
        for i in range(1, count + 1):
            problem = generate_problem(level, i)
            problems.append(problem)
            print(f"  {problem['id']}: {problem['topology']}")
    
    # Save
    output_file = BENCHMARK_DIR / "test_problems_v2.json"
    with open(output_file, "w") as f:
        json.dump({
            "version": "2.0",
            "description": "PowerElecBench Test Set v2 - ALL TOPOLOGIES",
            "generated": datetime.now().isoformat(),
            "distribution": distribution,
            "total": len(problems),
            "problems": problems
        }, f, indent=2)
    
    print(f"\n✅ Generated {len(problems)} problems")
    print(f"Saved to: {output_file}")
    
    # Print topology distribution
    print("\n=== Topology Distribution ===")
    for level in [1, 2, 3, 4]:
        level_problems = [p for p in problems if p["level"] == level]
        topos = {}
        for p in level_problems:
            t = p["topology"]
            topos[t] = topos.get(t, 0) + 1
        print(f"L{level}: {topos}")


if __name__ == "__main__":
    main()
