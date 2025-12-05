#!/usr/bin/env python3
"""
Deep dive into SPICE topology issues
Test each topology individually with known-good parameters to identify problems
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from spice_evaluator import PowerConverterEvaluator, SpiceNetlistGenerator

def test_topology(topology: str, vin: float, vout_target: float, duty: float, 
                  power: float = 100, fsw: float = 100e3, n: float = 1.0):
    """Test a single topology with given parameters"""
    
    evaluator = PowerConverterEvaluator()
    
    # Calculate load resistance
    R_load = vout_target**2 / power if power > 0 else 10
    
    # Calculate L and C (reasonable values)
    L = 100e-6  # 100µH default
    C = 100e-6  # 100µF default
    
    components = {
        "D": duty,
        "L": L,
        "C": C,
        "R_load": R_load
    }
    
    specs = {
        "vin": vin,
        "vout": abs(vout_target),  # Use absolute value for target
        "power": power,
        "fsw": fsw,
        "n": n  # Include turns ratio for isolated topologies
    }
    
    print(f"\n{'='*60}")
    print(f"TOPOLOGY: {topology.upper()}")
    print(f"{'='*60}")
    print(f"Vin={vin}V, Vout_target={vout_target}V, D={duty}, P={power}W")
    print(f"L={L*1e6:.1f}µH, C={C*1e6:.1f}µF, R_load={R_load:.1f}Ω")
    
    # Run simulation
    try:
        results, score = evaluator.evaluate(topology, components, specs, level=3)
        
        print(f"\n--- RESULTS ---")
        print(f"Simulation: {'SUCCESS' if results.simulation_success else 'FAILED'}")
        if results.error_message:
            print(f"Error: {results.error_message}")
        
        print(f"Vout DC: {results.vout_dc:.2f}V (target: {vout_target}V)")
        vout_error = abs(results.vout_dc - abs(vout_target)) / abs(vout_target) * 100 if vout_target != 0 else 0
        print(f"Vout Error: {vout_error:.1f}%")
        print(f"Ripple: {results.vout_ripple_pct*100:.2f}%")
        print(f"Efficiency: {results.efficiency*100:.0f}%")
        print(f"IL avg: {results.il_avg:.2f}A, IL_min: {results.il_min:.2f}A")
        
        print(f"\n--- SCORES ---")
        print(f"Total: {score.total_score:.1f}/100")
        print(f"  Vout: {score.vout_score:.1f}")
        print(f"  Ripple: {score.ripple_score:.1f}")
        print(f"  Efficiency: {score.efficiency_score:.1f}")
        print(f"  Current: {score.current_score:.1f}")
        print(f"  Stress: {score.stress_score:.1f}")
        
        return {
            "success": results.simulation_success,
            "vout": results.vout_dc,
            "vout_error_pct": vout_error,
            "score": score.total_score
        }
        
    except Exception as e:
        print(f"EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


def main():
    print("="*70)
    print("SPICE TOPOLOGY DIAGNOSTIC TEST")
    print("="*70)
    
    # Test cases with KNOWN correct Vout equations
    # Format: (topology, vin, vout_target, duty, power, fsw, n)
    test_cases = [
        # BUCK: Vout = Vin * D
        ("buck", 24, 12, 0.5, 100, 100e3, 1.0),  # 24*0.5 = 12V ✓
        ("buck", 48, 12, 0.25, 100, 100e3, 1.0), # 48*0.25 = 12V ✓
        
        # BOOST: Vout = Vin / (1-D)
        ("boost", 12, 24, 0.5, 100, 100e3, 1.0),  # 12/(1-0.5) = 24V ✓
        ("boost", 12, 36, 0.667, 100, 100e3, 1.0), # 12/(1-0.667) = 36V ✓
        
        # BUCK-BOOST: Vout = -Vin * D / (1-D)
        ("buck_boost", 24, -24, 0.5, 100, 100e3, 1.0),  # -24*0.5/0.5 = -24V ✓
        ("buck_boost", 24, -12, 0.333, 100, 100e3, 1.0), # -24*0.333/0.667 = -12V ✓
        
        # SEPIC: Vout = Vin * D / (1-D)  (same as buck-boost magnitude, but positive)
        ("sepic", 24, 24, 0.5, 100, 100e3, 1.0),  # 24*0.5/0.5 = 24V ✓
        ("sepic", 12, 24, 0.667, 100, 100e3, 1.0), # 12*0.667/0.333 = 24V ✓
        
        # CUK: Vout = -Vin * D / (1-D) (same as buck-boost, inverting)
        ("cuk", 24, -24, 0.5, 100, 100e3, 1.0),
        
        # FLYBACK: Vout = Vin * D * N / (1-D), SPICE model uses N=1
        ("flyback", 100, 100, 0.5, 100, 100e3, 1.0),  # 100*0.5*1/0.5 = 100V ✓
        ("flyback", 200, 100, 0.333, 100, 100e3, 1.0), # 200*0.333*1/0.667 = 100V ✓
        
        # FORWARD: Vout = Vin * D * N, SPICE model uses N=0.5
        ("forward", 200, 50, 0.5, 100, 100e3, 0.5),  # 200*0.5*0.5 = 50V ✓
        ("forward", 400, 48, 0.24, 100, 100e3, 0.5), # 400*0.24*0.5 = 48V ✓
        
        # HALF-BRIDGE: Vout = Vin * D * N, SPICE model uses N=0.25
        ("half_bridge", 400, 24, 0.24, 100, 100e3, 0.25),  # 400*0.24*0.25 = 24V
        
        # FULL-BRIDGE: Vout = Vin * D * N, SPICE model uses N=0.25
        ("full_bridge", 400, 48, 0.48, 100, 100e3, 0.25),  # 400*0.48*0.25 = 48V
        
        # PUSH-PULL: Vout = Vin * D * N, SPICE model uses N=0.5
        ("push_pull", 48, 24, 0.5, 100, 100e3, 0.5),  # 48*0.5*0.5*2 = 24V (2x freq)
    ]
    
    results_summary = []
    
    for test in test_cases:
        result = test_topology(*test)
        results_summary.append({
            "topology": test[0],
            "vin": test[1],
            "vout_target": test[2],
            "duty": test[3],
            **result
        })
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Topology':<15} {'Vin':>6} {'Target':>8} {'D':>6} {'Vout':>8} {'Error%':>8} {'Score':>8}")
    print("-"*70)
    
    for r in results_summary:
        success = "✓" if r.get("success") and r.get("vout_error_pct", 100) < 10 else "✗"
        vout = f"{r.get('vout', 0):.1f}V" if r.get("vout") else "N/A"
        error = f"{r.get('vout_error_pct', 0):.1f}%" if r.get("vout_error_pct") is not None else "N/A"
        score = f"{r.get('score', 0):.0f}" if r.get("score") else "N/A"
        print(f"{r['topology']:<15} {r['vin']:>5}V {r['vout_target']:>7}V {r['duty']:>6.2f} {vout:>8} {error:>8} {score:>8} {success}")
    
    # Count working topologies
    working = sum(1 for r in results_summary if r.get("success") and r.get("vout_error_pct", 100) < 20)
    print(f"\nWorking topologies (< 20% error): {working}/{len(results_summary)}")


if __name__ == "__main__":
    main()
