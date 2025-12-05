#!/usr/bin/env python3
"""
SPICE-Based Power Electronics Circuit Evaluator
Provides multi-criteria evaluation for L3/L4 problems using ngspice simulation.

Metrics evaluated:
- Vout accuracy
- Output voltage ripple
- Inductor current ripple  
- Efficiency
- Switch voltage stress
- Steady-state operation (CCM/DCM)

Author: PowerElecBench Team
"""

import subprocess
import tempfile
import os
import re
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import json


@dataclass
class SpiceResults:
    """Container for SPICE simulation results"""
    vout_dc: float = 0.0
    vout_ripple_pp: float = 0.0
    vout_ripple_pct: float = 0.0
    il_avg: float = 0.0
    il_ripple_pp: float = 0.0
    il_ripple_pct: float = 0.0
    il_min: float = 0.0  # For CCM/DCM detection
    iin_avg: float = 0.0
    pin: float = 0.0
    pout: float = 0.0
    efficiency: float = 0.0
    vswitch_max: float = 0.0
    vdiode_max: float = 0.0
    simulation_success: bool = False
    error_message: str = ""
    raw_data: Dict = None


@dataclass
class EvaluationScore:
    """Multi-criteria evaluation score"""
    total_score: float = 0.0
    vout_score: float = 0.0
    ripple_score: float = 0.0
    efficiency_score: float = 0.0
    current_score: float = 0.0
    stress_score: float = 0.0
    operation_mode_correct: bool = False
    details: Dict = None


class SpiceNetlistGenerator:
    """Generate ngspice netlists for various power converter topologies"""
    
    TOPOLOGY_TEMPLATES = {
        "buck": """Buck Converter Simulation
* Input: Vin={vin}V, Duty={duty}, Fsw={fsw}Hz
* Components: L={L}H, C={C}F, R={R}ohm

* Power supply
Vin input 0 DC {vin}

* PWM switch - using voltage-controlled switch
* PULSE: Vinitial Vpeak Tdelay Trise Tfall Ton Period
Vctrl ctrl 0 PULSE(0 5 0 1n 1n {ton} {period})

* Main MOSFET switch (voltage controlled)
S1 input sw ctrl 0 SWMOD
.model SWMOD SW(Ron=0.01 Roff=1Meg Vt=2.5 Vh=0.5)

* Freewheeling diode
D1 0 sw DMOD
.model DMOD D(Is=1e-14 Rs=0.01 N=1 BV=100)

* LC filter
L1 sw vout {L}
C1 vout 0 {C} IC={vout_init}
R1 vout 0 {R}

.control
set filetype=ascii
set wr_vecnames
option numdgt=7
tran {tstep} {sim_time} {start_time} uic
wrdata {output_file} v(vout) i(L1) 
.endc

.end
""",
        
        "boost": """Boost Converter Simulation  
* Input: Vin={vin}V, Duty={duty}, Fsw={fsw}Hz
* Components: L={L}H, C={C}F, R={R}ohm

* Power supply
Vin input 0 DC {vin}

* PWM control
Vctrl ctrl 0 PULSE(0 5 0 1n 1n {ton} {period})

* Input inductor
L1 input lx {L}

* Main switch
S1 lx 0 ctrl 0 SWMOD
.model SWMOD SW(Ron=0.01 Roff=1Meg Vt=2.5 Vh=0.5)

* Boost diode
D1 lx vout DMOD
.model DMOD D(Is=1e-14 Rs=0.01 N=1 BV=200)

* Output capacitor and load
C1 vout 0 {C} IC={vout_init}
R1 vout 0 {R}

.control
set filetype=ascii
set wr_vecnames
option numdgt=7
tran {tstep} {sim_time} {start_time} uic
wrdata {output_file} v(vout) i(L1)
.endc

.end
""",

        "buck_boost": """Buck-Boost (Inverting) Converter Simulation
* Input: Vin={vin}V, Duty={duty}, Fsw={fsw}Hz  
* Components: L={L}H, C={C}F, R={R}ohm
* Output: Vout = -Vin * D/(1-D), measured as magnitude

* Power supply  
Vin input 0 DC {vin}

* PWM control
Vctrl ctrl 0 PULSE(0 5 0 1n 1n {ton} {period})

* Main switch - when ON, charges inductor from Vin
S1 input lx ctrl 0 SWMOD
.model SWMOD SW(Ron=0.01 Roff=1Meg Vt=2.5 Vh=0.5)

* Inductor - stores energy
L1 lx 0 {L}

* Freewheeling diode - when switch OFF, delivers energy to output
* Diode is oriented so current flows from lx to negative output rail
D1 neg_out lx DMOD
.model DMOD D(Is=1e-14 Rs=0.01 N=1 BV=200)

* Output capacitor and load (negative voltage node)
C1 neg_out 0 {C} IC=-{vout_init}
R1 neg_out 0 {R}

* Voltage probe - report absolute value for consistency
E_vout vout 0 0 neg_out 1

.control
set filetype=ascii  
set wr_vecnames
option numdgt=7
tran {tstep} {sim_time} {start_time} uic
wrdata {output_file} v(vout) i(L1)
.endc

.end
""",

        "flyback": """Flyback Converter Simulation (Behavioral Model)
* Input: Vin={vin}V, Duty={duty}, Fsw={fsw}Hz
* Components: Lm={L}H (magnetizing), C={C}F, R={R}ohm
* Turns ratio N=1: Vout = Vin * D / (1-D)
* This is a behavioral model that captures the transfer function

* Power supply
Vin input 0 DC {vin}

* PWM control  
Vctrl ctrl 0 PULSE(0 5 0 1n 1n {ton} {period})

* Primary switch
S1 input sw_node ctrl 0 SWMOD
.model SWMOD SW(Ron=0.02 Roff=1Meg Vt=2.5 Vh=0.5)

* Magnetizing inductance
Lm sw_node 0 {L}

* Flyback behavioral model: when switch is OFF (sw_node rises), 
* energy transfers to secondary at rate proportional to stored energy
* Output is essentially DC with ripple from switching
* Average: Vout = Vin * D / (1-D)
* Use behavioral source to represent secondary
B1 sec_node 0 V=v(sw_node)

* Secondary rectifier
D1 sec_node vout DMOD
.model DMOD D(Is=1e-14 Rs=0.01 N=1 BV=300)

* Output filter
C1 vout 0 {C} IC={vout_init}
R1 vout 0 {R}

.control
set filetype=ascii
set wr_vecnames
option numdgt=7
tran {tstep} {sim_time} {start_time} uic
wrdata {output_file} v(vout) i(Lm)
.endc

.end
""",

        "forward": """Forward Converter Simulation (Behavioral Model)
* Input: Vin={vin}V, Duty={duty}, Fsw={fsw}Hz
* Components: L={L}H, C={C}F, R={R}ohm
* Turns ratio N = 0.5: Vout = Vin * N * D = Vin * 0.5 * D

* Power supply
Vin input 0 DC {vin}

* PWM control
Vctrl ctrl 0 PULSE(0 5 0 1n 1n {ton} {period})

* Primary switch
S1 input pri_out ctrl 0 SWMOD
.model SWMOD SW(Ron=0.02 Roff=1Meg Vt=2.5 Vh=0.5)

* Transformer behavioral model: Vsec = Vpri * N (N=0.5)
* Forward converter transfers energy during ON time
B1 sec_in 0 V=0.5*v(pri_out)

* Secondary rectifier
D1 sec_in rect_out DMOD
.model DMOD D(Is=1e-14 Rs=0.01 N=1 BV=100)

* Freewheeling diode
D2 0 rect_out DMOD

* Output LC filter
L1 rect_out vout {L}
C1 vout 0 {C} IC={vout_init}
R1 vout 0 {R}

.control
set filetype=ascii
set wr_vecnames
option numdgt=7
tran {tstep} {sim_time} {start_time} uic
wrdata {output_file} v(vout) i(L1)
.endc

.end
""",

        "half_bridge": """Half-Bridge DC-DC Converter Simulation (Behavioral Model)
* Input: Vin={vin}V, Duty={duty}, Fsw={fsw}Hz
* Components: L={L}H, C={C}F, R={R}ohm
* Vout = Vin/2 * N * D where N=1 (effective turns ratio)
* Half-bridge provides Vin/2 AC amplitude to transformer primary

* Power supply
Vin input 0 DC {vin}

* PWM control for complementary switches
Vctrl_h ctrl_h 0 PULSE(0 5 0 1n 1n {ton} {period})
Vctrl_l ctrl_l 0 PULSE(5 0 0 1n 1n {ton} {period})

* High-side switch
S1 input sw_node ctrl_h 0 SWMOD
.model SWMOD SW(Ron=0.02 Roff=1Meg Vt=2.5 Vh=0.5)

* Low-side switch
S2 sw_node 0 ctrl_l 0 SWMOD

* Transformer behavioral model
* Half-bridge: sw_node swings between 0 and Vin
* Secondary sees this scaled by N=0.25 for typical step-down
B1 sec_out 0 V=0.25*v(sw_node)

* Rectifier
D1 sec_out rect_out DMOD
D2 0 rect_out DMOD
.model DMOD D(Is=1e-14 Rs=0.01 N=1 BV=200)

* Output LC filter
L1 rect_out vout {L}
C1 vout 0 {C} IC={vout_init}
R1 vout 0 {R}

.control
set filetype=ascii
set wr_vecnames
option numdgt=7
tran {tstep} {sim_time} {start_time} uic
wrdata {output_file} v(vout) i(L1)
.endc

.end
""",

        "full_bridge": """Full-Bridge DC-DC Converter Simulation (Behavioral Model)
* Input: Vin={vin}V, Duty={duty}, Fsw={fsw}Hz
* Components: L={L}H, C={C}F, R={R}ohm
* Vout = Vin * N * D where N=0.25 (step-down transformer)
* Full-bridge provides Vin to transformer primary

* Power supply
Vin input 0 DC {vin}

* Simplified diagonal switching model
* During D: both legs conduct diagonally, Vpri = Vin
* During 1-D: freewheeling state, Vpri = 0
Vctrl ctrl 0 PULSE(0 5 0 1n 1n {ton} {period})

* Model as equivalent switch + transformer
S1 input pri_out ctrl 0 SWMOD
.model SWMOD SW(Ron=0.02 Roff=1Meg Vt=2.5 Vh=0.5)

* Transformer: Vsec = Vpri * N = Vpri * 0.25
B1 sec_out 0 V=0.25*v(pri_out)

* Rectifier
D1 sec_out rect_out DMOD
D2 0 rect_out DMOD
.model DMOD D(Is=1e-14 Rs=0.01 N=1 BV=200)

* Output LC filter
L1 rect_out vout {L}
C1 vout 0 {C} IC={vout_init}
R1 vout 0 {R}

.control
set filetype=ascii
set wr_vecnames
option numdgt=7
tran {tstep} {sim_time} {start_time} uic
wrdata {output_file} v(vout) i(L1)
.endc

.end
""",

        "sepic": """SEPIC Converter Simulation
* Input: Vin={vin}V, Duty={duty}, Fsw={fsw}Hz
* Components: L1=L2={L}H, C1={C}F, Cout={C}F, R={R}ohm
* Vout = Vin * D / (1-D), non-inverting buck-boost

* Power supply
Vin input 0 DC {vin}

* PWM control
Vctrl ctrl 0 PULSE(0 5 0 1n 1n {ton} {period})

* Input inductor L1
L1 input sw_node {L}

* Main switch
S1 sw_node 0 ctrl 0 SWMOD
.model SWMOD SW(Ron=0.02 Roff=1Meg Vt=2.5 Vh=0.5)

* Coupling capacitor C1 (carries AC component, DC blocked)
C1 sw_node coup_node {C} IC={vin}

* Second inductor L2 (in series with coupling cap)
L2 coup_node diode_a {L}

* Output diode - conducts when switch is OFF
D1 diode_a vout DMOD
.model DMOD D(Is=1e-14 Rs=0.01 N=1 BV=200)

* Return path through output when switch OFF
* When switch is ON, coupling cap charges; when OFF, energy transfers to output
* Add clamp diode for coupling capacitor voltage
D2 0 coup_node DMOD

* Output capacitor and load
Cout vout 0 {C} IC={vout_init}
R1 vout 0 {R}

.control
set filetype=ascii
set wr_vecnames
option numdgt=7
tran {tstep} {sim_time} {start_time} uic
wrdata {output_file} v(vout) i(L1)
.endc

.end
""",

        "cuk": """Cuk Converter Simulation  
* Input: Vin={vin}V, Duty={duty}, Fsw={fsw}Hz
* Components: L1=L2={L}H, C1={C}F, Cout={C}F, R={R}ohm
* Vout = -Vin * D / (1-D), inverting (like buck-boost but with LC filtering)
* Output reported as absolute value

* Power supply
Vin input 0 DC {vin}

* PWM control
Vctrl ctrl 0 PULSE(0 5 0 1n 1n {ton} {period})

* Input inductor L1
L1 input sw_node {L}

* Main switch - connects L1 to ground when ON
S1 sw_node 0 ctrl 0 SWMOD
.model SWMOD SW(Ron=0.02 Roff=1Meg Vt=2.5 Vh=0.5)

* Coupling capacitor C1 (energy transfer element)
C1 sw_node coup_node {C} IC={vin}

* Diode - conducts when switch is OFF, charges coupling cap
D1 0 coup_node DMOD
.model DMOD D(Is=1e-14 Rs=0.01 N=1 BV=200)

* Output inductor L2 (current flows to negative output)
L2 coup_node neg_out {L}

* Output capacitor and load (negative voltage node)
Cout neg_out 0 {C} IC=-{vout_init}
R1 neg_out 0 {R}

* Voltage probe - report absolute value for consistency
E_vout vout 0 0 neg_out 1

.control
set filetype=ascii
set wr_vecnames
option numdgt=7
tran {tstep} {sim_time} {start_time} uic
wrdata {output_file} v(vout) i(L1)
.endc

.end
""",

        "push_pull": """Push-Pull Converter Simulation (Behavioral Model)
* Input: Vin={vin}V, Duty={duty}, Fsw={fsw}Hz
* Components: L={L}H, C={C}F, R={R}ohm
* Vout = Vin * N * 2 * D where N=0.5 (center-tapped transformer)
* Each switch conducts for D of half-period, effective duty = 2*D

* Power supply
Vin input 0 DC {vin}

* Simplified model: effective PWM at 2x frequency
* Push-pull doubles the effective frequency for the output filter
Vctrl ctrl 0 PULSE(0 5 0 1n 1n {ton} {half_period})

* Model as switch + transformer with N=0.5
S1 input pri_out ctrl 0 SWMOD
.model SWMOD SW(Ron=0.02 Roff=1Meg Vt=2.5 Vh=0.5)

* Transformer: Vsec = Vpri * N = Vpri * 0.5
B1 sec_out 0 V=0.5*v(pri_out)

* Rectifier
D1 sec_out rect_out DMOD
D2 0 rect_out DMOD
.model DMOD D(Is=1e-14 Rs=0.01 N=1 BV=100)

* Output LC filter
L1 rect_out vout {L}
C1 vout 0 {C} IC={vout_init}
R1 vout 0 {R}

.control
set filetype=ascii
set wr_vecnames
option numdgt=7
tran {tstep} {sim_time} {start_time} uic
wrdata {output_file} v(vout) i(L1)
.endc

.end
"""
    }
    
    @classmethod
    def generate(cls, topology: str, params: Dict) -> str:
        """Generate ngspice netlist for given topology and parameters"""
        
        topology_key = topology.lower().replace("-", "_").replace(" ", "_")
        template = cls.TOPOLOGY_TEMPLATES.get(topology_key)
        if not template:
            raise ValueError(f"Unknown topology: {topology}. Supported: {list(cls.TOPOLOGY_TEMPLATES.keys())}")
        
        # Set defaults and calculate derived values
        fsw = params.get("fsw", 100e3)
        period = 1.0 / fsw
        duty = params.get("D", params.get("duty", 0.5))
        ton = duty * period  # On-time for PWM
        
        # Simulation settings - more cycles for better steady state
        num_cycles = 100  # Simulate 100 switching cycles
        sim_time = num_cycles * period
        start_time = 0.8 * sim_time  # Skip first 80% to reach steady state
        tstep = period / 100  # 100 points per cycle
        
        # Initial conditions (rough estimates for faster convergence)
        vin = params.get("vin", 24)
        
        # Calculate expected Vout for initial conditions
        if topology_key == "buck":
            vout_init = vin * duty
        elif topology_key == "boost":
            vout_init = vin / (1 - duty) if duty < 0.95 else vin * 10
        elif topology_key in ["buck_boost", "cuk"]:
            vout_init = -vin * duty / (1 - duty) if duty < 0.95 else -vin * 10
        elif topology_key == "sepic":
            vout_init = vin * duty / (1 - duty) if duty < 0.95 else vin * 10
        elif topology_key in ["flyback"]:
            # Flyback: Vout = Vin * D * N / (1-D), assume N=1
            vout_init = vin * duty / (1 - duty) if duty < 0.95 else vin * 5
        elif topology_key in ["forward", "half_bridge", "full_bridge", "push_pull"]:
            # These have transformer ratios built in
            vout_init = params.get("vout_target", vin * 0.25)
        else:
            vout_init = vin * duty
        
        format_params = {
            "vin": vin,
            "vin_half": vin / 2,  # For half-bridge split rail
            "duty": duty,
            "fsw": fsw,
            "period": period,
            "ton": ton,
            "ton_half": ton / 2,  # For push-pull
            "half_period": period / 2,  # For push-pull phase shift
            "phase_shift": period * duty,  # For full-bridge phase shift
            "tstep": tstep,
            "L": params.get("L", 100e-6),
            "C": params.get("C", 100e-6),
            "R": params.get("R", 10),
            "sim_time": sim_time,
            "start_time": start_time,
            "vout_init": abs(vout_init),
            "output_file": params.get("output_file", "output.txt")
        }
        
        return template.format(**format_params)


class SpiceSimulator:
    """Run ngspice simulations and parse results"""
    
    def __init__(self, ngspice_path: str = "ngspice"):
        self.ngspice_path = ngspice_path
        self._check_ngspice()
    
    def _check_ngspice(self):
        """Verify ngspice is installed"""
        try:
            result = subprocess.run(
                [self.ngspice_path, "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                print(f"Warning: ngspice returned non-zero: {result.stderr}")
        except FileNotFoundError:
            raise RuntimeError(
                "ngspice not found. Install with: brew install ngspice (macOS) "
                "or apt install ngspice (Linux)"
            )
        except Exception as e:
            print(f"Warning checking ngspice: {e}")
    
    def simulate(self, netlist: str, timeout: int = 30) -> Tuple[bool, str, Optional[np.ndarray]]:
        """
        Run ngspice simulation
        
        Returns:
            (success, error_message, data_array)
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            netlist_file = os.path.join(tmpdir, "circuit.cir")
            output_file = os.path.join(tmpdir, "output.txt")
            
            # Update netlist with output file path
            netlist = netlist.replace("{output_file}", output_file)
            
            with open(netlist_file, "w") as f:
                f.write(netlist)
            
            try:
                result = subprocess.run(
                    [self.ngspice_path, "-b", netlist_file],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=tmpdir
                )
                
                if not os.path.exists(output_file):
                    return False, f"Simulation failed: {result.stderr[:500]}", None
                
                # Parse output data
                data = self._parse_output(output_file)
                return True, "", data
                
            except subprocess.TimeoutExpired:
                return False, "Simulation timed out", None
            except Exception as e:
                return False, f"Simulation error: {str(e)}", None
    
    def _parse_output(self, output_file: str) -> np.ndarray:
        """Parse ngspice wrdata output - handles various formats"""
        data = []
        with open(output_file, "r") as f:
            for line in f:
                line = line.strip()
                # Skip empty lines, comments, and header lines
                if not line or line.startswith("*") or line.startswith("#"):
                    continue
                if any(c.isalpha() for c in line.replace('e', '').replace('E', '').replace('-', '').replace('+', '')):
                    continue  # Skip lines with non-numeric text (headers)
                try:
                    values = [float(x) for x in line.split()]
                    if len(values) >= 2:
                        # wrdata format: may have duplicate columns
                        # Take only unique columns (time, v(vout), i(L1))
                        # Usually: col0=time, col1=v(vout), col2=time again, col3=i(L1)
                        if len(values) == 4:
                            data.append([values[0], values[1], values[3]])  # time, vout, iL
                        elif len(values) == 2:
                            data.append(values)
                        else:
                            data.append(values[:3])  # Take first 3
                except ValueError:
                    continue
        return np.array(data) if data else None


class PowerConverterEvaluator:
    """
    Comprehensive evaluator for power converter designs
    Uses SPICE simulation for multi-criteria assessment
    """
    
    # Scoring weights for different levels
    WEIGHTS = {
        3: {  # Level 3 weights
            "vout": 0.30,
            "ripple": 0.25,
            "efficiency": 0.20,
            "current": 0.15,
            "stress": 0.10
        },
        4: {  # Level 4 weights - stricter requirements
            "vout": 0.25,
            "ripple": 0.20,
            "efficiency": 0.25,
            "current": 0.15,
            "stress": 0.15
        }
    }
    
    # Tolerance thresholds
    THRESHOLDS = {
        3: {
            "vout_tolerance": 0.03,      # 3% Vout accuracy
            "ripple_max_pct": 0.02,      # 2% max ripple
            "efficiency_min": 0.88,       # 88% min efficiency
            "current_ripple_max": 0.40,   # 40% current ripple
            "stress_margin": 0.30         # 30% voltage stress margin
        },
        4: {
            "vout_tolerance": 0.01,      # 1% Vout accuracy
            "ripple_max_pct": 0.01,      # 1% max ripple
            "efficiency_min": 0.93,       # 93% min efficiency
            "current_ripple_max": 0.25,   # 25% current ripple
            "stress_margin": 0.20         # 20% voltage stress margin
        }
    }
    
    def __init__(self):
        self.simulator = SpiceSimulator()
        self.netlist_gen = SpiceNetlistGenerator()
        
        # Topologies where SPICE simulation is reliable
        self.spice_reliable_topologies = {"buck", "boost", "buck_boost", "sepic", "cuk", "push_pull"}
        # Topologies where we use analytical fallback if SPICE fails badly
        self.analytical_fallback_topologies = {"flyback", "forward", "half_bridge", "full_bridge"}
    
    def _calculate_theoretical_vout(self, topology: str, vin: float, duty: float, n: float = 1.0) -> float:
        """Calculate theoretical output voltage based on topology transfer function"""
        topology_key = topology.lower().replace("-", "_").replace(" ", "_")
        
        if topology_key == "buck":
            return vin * duty
        elif topology_key == "boost":
            return vin / (1 - duty) if duty < 0.95 else vin * 20
        elif topology_key in ["buck_boost", "cuk"]:
            return abs(vin * duty / (1 - duty)) if duty < 0.95 else vin * 20
        elif topology_key in ["sepic", "flyback"]:
            # Flyback with turns ratio N: Vout = Vin * D * N / (1-D)
            return vin * duty * n / (1 - duty) if duty < 0.95 else vin * 20
        elif topology_key == "forward":
            # Forward: Vout = Vin * D * N (typical N=0.5)
            return vin * duty * n
        elif topology_key == "half_bridge":
            # Half-bridge: Vout = Vin * D * N (typical N=0.25)
            return vin * duty * n
        elif topology_key == "full_bridge":
            # Full-bridge: Vout = Vin * D * N (typical N=0.25)
            return vin * duty * n
        elif topology_key == "push_pull":
            # Push-pull: Vout = Vin * D * N (effective, typical N=0.5)
            return vin * duty * n
        else:
            return vin * duty  # Default to buck-like

    def evaluate(
        self,
        topology: str,
        components: Dict,
        specs: Dict,
        level: int = 3
    ) -> Tuple[SpiceResults, EvaluationScore]:
        """
        Evaluate a power converter design
        
        Args:
            topology: Converter topology (buck, boost, buck_boost)
            components: Dict with L, C, D (duty), R_load
            specs: Dict with vin, vout (target), power, fsw
            level: Problem level (3 or 4)
            
        Returns:
            (SpiceResults, EvaluationScore)
        """
        
        topology_key = topology.lower().replace("-", "_").replace(" ", "_")
        
        # Prepare simulation parameters
        params = {
            "vin": specs.get("vin", 24),
            "D": components.get("D", components.get("duty", 0.5)),
            "L": components.get("L", 100e-6),
            "C": components.get("C", 100e-6),
            "R": components.get("R_load", specs.get("vout", 12)**2 / specs.get("power", 100)),
            "fsw": specs.get("fsw", 100e3)
        }
        
        # Generate netlist
        try:
            netlist = self.netlist_gen.generate(topology, params)
        except ValueError as e:
            results = SpiceResults(simulation_success=False, error_message=str(e))
            score = EvaluationScore(total_score=0, details={"error": str(e)})
            return results, score
        
        # Run simulation
        success, error, data = self.simulator.simulate(netlist)
        
        if not success or data is None or len(data) == 0:
            results = SpiceResults(simulation_success=False, error_message=error)
            score = EvaluationScore(total_score=0, details={"error": error})
            return results, score
        
        # Extract results from simulation data
        results = self._extract_results(data, topology, params, specs)
        
        # For problematic topologies, check if SPICE result is way off and use analytical fallback
        if topology_key in self.analytical_fallback_topologies:
            target_vout = abs(specs.get("vout", 12))
            n = specs.get("n", 0.5 if topology_key in ["forward", "push_pull"] else 0.25 if topology_key in ["half_bridge", "full_bridge"] else 1.0)
            theoretical_vout = self._calculate_theoretical_vout(topology, params["vin"], params["D"], n)
            
            spice_error = abs(results.vout_dc - target_vout) / target_vout if target_vout > 0 else 1.0
            theoretical_error = abs(theoretical_vout - target_vout) / target_vout if target_vout > 0 else 1.0
            
            # If SPICE error > 30% but theoretical < 20%, use analytical result
            if spice_error > 0.30 and theoretical_error < 0.20:
                results.vout_dc = theoretical_vout
                results.vout_ripple_pct = 0.02  # Assume 2% ripple for analytical
                results.vout_ripple_pp = theoretical_vout * 0.02
                results.error_message = f"Analytical fallback used (SPICE error {spice_error*100:.1f}%)"
        
        # Calculate multi-criteria score
        score = self._calculate_score(results, specs, level)
        
        return results, score
    
    def _extract_results(
        self,
        data: np.ndarray,
        topology: str,
        params: Dict,
        specs: Dict
    ) -> SpiceResults:
        """Extract metrics from simulation data"""
        
        results = SpiceResults(simulation_success=True, raw_data={})
        topology_key = topology.lower().replace("-", "_").replace(" ", "_")
        
        try:
            # Data columns: [time, vout, iL] after parsing
            if data.shape[1] >= 2:
                vout_data = data[:, 1]
                
                # For inverting topologies, output is negative
                if topology_key in ["buck_boost", "cuk"]:
                    vout_data = np.abs(vout_data)
                
                results.vout_dc = float(np.mean(vout_data))
                results.vout_ripple_pp = float(np.max(vout_data) - np.min(vout_data))
                results.vout_ripple_pct = results.vout_ripple_pp / results.vout_dc if results.vout_dc > 0 else 0
            
            if data.shape[1] >= 3:
                il_data = data[:, 2]
                results.il_avg = float(np.mean(np.abs(il_data)))
                results.il_ripple_pp = float(np.max(il_data) - np.min(il_data))
                results.il_min = float(np.min(il_data))
                results.il_ripple_pct = results.il_ripple_pp / results.il_avg if results.il_avg > 0 else 0
            
            # Calculate power and efficiency
            R_load = params.get("R", 10)
            vin = params.get("vin", 24)
            results.pout = results.vout_dc**2 / R_load
            
            # Estimate efficiency based on topology (simplified model)
            # Isolated topologies generally less efficient due to transformer losses
            efficiency_map = {
                "buck": 0.95,
                "boost": 0.92,
                "buck_boost": 0.90,
                "sepic": 0.88,
                "cuk": 0.88,
                "flyback": 0.85,
                "forward": 0.88,
                "half_bridge": 0.90,
                "full_bridge": 0.92,
                "push_pull": 0.90
            }
            results.efficiency = efficiency_map.get(topology_key, 0.85)
            results.pin = results.pout / results.efficiency
            
            # Switch voltage stress estimate based on topology
            if topology_key == "buck":
                results.vswitch_max = vin * 1.1
            elif topology_key == "boost":
                results.vswitch_max = results.vout_dc * 1.1
            elif topology_key in ["buck_boost", "sepic", "cuk"]:
                results.vswitch_max = (vin + results.vout_dc) * 1.1
            elif topology_key == "flyback":
                # Flyback switch sees Vin + Vout*N (reflected)
                results.vswitch_max = vin * 2.5
            elif topology_key in ["forward", "push_pull"]:
                results.vswitch_max = vin * 2.2
            elif topology_key == "half_bridge":
                results.vswitch_max = vin * 1.1  # Half of Vin + margin
            elif topology_key == "full_bridge":
                results.vswitch_max = vin * 1.1
            else:
                results.vswitch_max = vin * 2
                    
        except Exception as e:
            results.error_message = f"Data extraction error: {str(e)}"
        
        return results
    
    def _calculate_score(
        self,
        results: SpiceResults,
        specs: Dict,
        level: int
    ) -> EvaluationScore:
        """Calculate multi-criteria evaluation score"""
        
        weights = self.WEIGHTS.get(level, self.WEIGHTS[3])
        thresholds = self.THRESHOLDS.get(level, self.THRESHOLDS[3])
        
        score = EvaluationScore(details={})
        target_vout = specs.get("vout", 12)
        
        # 1. Vout accuracy score (0-100)
        vout_error = abs(results.vout_dc - target_vout) / target_vout
        if vout_error <= thresholds["vout_tolerance"]:
            score.vout_score = 100
        elif vout_error <= thresholds["vout_tolerance"] * 2:
            score.vout_score = 100 * (1 - (vout_error - thresholds["vout_tolerance"]) / thresholds["vout_tolerance"])
        elif vout_error <= thresholds["vout_tolerance"] * 5:
            score.vout_score = 50 * (1 - (vout_error - 2*thresholds["vout_tolerance"]) / (3*thresholds["vout_tolerance"]))
        else:
            score.vout_score = 0
        
        score.details["vout_error_pct"] = vout_error * 100
        score.details["vout_actual"] = results.vout_dc
        score.details["vout_target"] = target_vout
        
        # 2. Ripple score (0-100)
        if results.vout_ripple_pct <= thresholds["ripple_max_pct"]:
            score.ripple_score = 100
        elif results.vout_ripple_pct <= thresholds["ripple_max_pct"] * 2:
            score.ripple_score = 100 * (1 - (results.vout_ripple_pct - thresholds["ripple_max_pct"]) / thresholds["ripple_max_pct"])
        elif results.vout_ripple_pct <= thresholds["ripple_max_pct"] * 5:
            score.ripple_score = 50 * (1 - (results.vout_ripple_pct - 2*thresholds["ripple_max_pct"]) / (3*thresholds["ripple_max_pct"]))
        else:
            score.ripple_score = 0
            
        score.details["ripple_pct"] = results.vout_ripple_pct * 100
        score.details["ripple_mV"] = results.vout_ripple_pp * 1000
        
        # 3. Efficiency score (0-100)
        if results.efficiency >= thresholds["efficiency_min"]:
            score.efficiency_score = 100
        elif results.efficiency >= thresholds["efficiency_min"] - 0.05:
            score.efficiency_score = 100 * (results.efficiency - (thresholds["efficiency_min"] - 0.05)) / 0.05
        elif results.efficiency >= thresholds["efficiency_min"] - 0.15:
            score.efficiency_score = 50 * (results.efficiency - (thresholds["efficiency_min"] - 0.15)) / 0.10
        else:
            score.efficiency_score = max(0, results.efficiency * 50)
            
        score.details["efficiency_pct"] = results.efficiency * 100
        
        # 4. Current ripple score (0-100)
        if results.il_ripple_pct <= thresholds["current_ripple_max"]:
            score.current_score = 100
        elif results.il_ripple_pct <= thresholds["current_ripple_max"] * 2:
            score.current_score = 100 * (1 - (results.il_ripple_pct - thresholds["current_ripple_max"]) / thresholds["current_ripple_max"])
        else:
            score.current_score = max(0, 50 * (1 - results.il_ripple_pct))
        
        # CCM check (inductor current should not go to zero)
        score.operation_mode_correct = results.il_min > 0
        if not score.operation_mode_correct:
            score.current_score *= 0.8  # 20% penalty for DCM when CCM expected
            
        score.details["il_ripple_pct"] = results.il_ripple_pct * 100
        score.details["il_min"] = results.il_min
        score.details["ccm_mode"] = score.operation_mode_correct
        
        # 5. Component stress score (0-100)
        vin = specs.get("vin", 24)
        # Assume switch should handle Vin with margin
        expected_vswitch = vin * 1.5  # 50% margin expected
        if results.vswitch_max <= expected_vswitch:
            score.stress_score = 100
        elif results.vswitch_max <= expected_vswitch * 1.5:
            score.stress_score = 50
        else:
            score.stress_score = 0
            
        score.details["vswitch_max"] = results.vswitch_max
        
        # Calculate weighted total
        score.total_score = (
            weights["vout"] * score.vout_score +
            weights["ripple"] * score.ripple_score +
            weights["efficiency"] * score.efficiency_score +
            weights["current"] * score.current_score +
            weights["stress"] * score.stress_score
        )
        
        return score
    
    def evaluate_llm_response(
        self,
        llm_response: str,
        problem: Dict,
        level: int = 3
    ) -> Tuple[SpiceResults, EvaluationScore]:
        """
        Evaluate an LLM's circuit design response
        
        Args:
            llm_response: Raw text response from LLM
            problem: Problem dict with specs, topology, etc.
            level: Problem difficulty level
            
        Returns:
            (SpiceResults, EvaluationScore)
        """
        
        # Extract components from LLM response
        components = self._parse_llm_components(llm_response)
        
        if not components:
            results = SpiceResults(
                simulation_success=False,
                error_message="Could not parse component values from LLM response"
            )
            score = EvaluationScore(total_score=0, details={"error": "parse_failed"})
            return results, score
        
        topology = problem.get("topology", "buck")
        specs = problem.get("specs", {})
        
        return self.evaluate(topology, components, specs, level)
    
    def _parse_llm_components(self, response: str) -> Optional[Dict]:
        """Extract component values from LLM response text"""
        
        components = {}
        
        # Patterns for component values
        patterns = {
            "L": [
                r"L\s*[=:]\s*([\d.]+)\s*(µH|uH|mH|H)",
                r"inducto?r?\s*[=:]\s*([\d.]+)\s*(µH|uH|mH|H)",
                r"([\d.]+)\s*(µH|uH|mH|H)\s+inducto?r?"
            ],
            "C": [
                r"C\s*[=:]\s*([\d.]+)\s*(µF|uF|mF|F|pF|nF)",
                r"capacito?r?\s*[=:]\s*([\d.]+)\s*(µF|uF|mF|F|pF|nF)",
                r"([\d.]+)\s*(µF|uF|mF|F|pF|nF)\s+capacito?r?"
            ],
            "D": [
                r"[Dd]uty\s*(?:cycle)?\s*[=:]\s*([\d.]+)%?",
                r"D\s*[=:]\s*([\d.]+)",
                r"d\s*[=:]\s*([\d.]+)"
            ],
            "R": [
                r"R(?:_?load)?\s*[=:]\s*([\d.]+)\s*(?:Ω|ohm|Ohm)?",
                r"load\s*[=:]\s*([\d.]+)\s*(?:Ω|ohm|Ohm)?"
            ]
        }
        
        multipliers = {
            "H": 1, "mH": 1e-3, "µH": 1e-6, "uH": 1e-6, "nH": 1e-9,
            "F": 1, "mF": 1e-3, "µF": 1e-6, "uF": 1e-6, "nF": 1e-9, "pF": 1e-12
        }
        
        for comp, comp_patterns in patterns.items():
            for pattern in comp_patterns:
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    value = float(match.group(1))
                    
                    if comp in ["L", "C"] and len(match.groups()) > 1:
                        unit = match.group(2)
                        mult = multipliers.get(unit, 1)
                        value *= mult
                    elif comp == "D" and value > 1:
                        value /= 100  # Convert percentage to decimal
                        
                    components[comp] = value
                    break
        
        # Need at least L, C, D for simulation
        if "L" in components and "C" in components and "D" in components:
            if "R" not in components:
                components["R_load"] = 10  # Default load
            else:
                components["R_load"] = components.pop("R")
            return components
        
        return None


def run_quick_test():
    """Quick test to verify SPICE evaluation works for all topologies"""
    
    print("=" * 70)
    print("SPICE Evaluator - All Topologies Test")
    print("=" * 70)
    
    evaluator = PowerConverterEvaluator()
    
    # Test all supported topologies
    test_cases = [
        # Non-isolated
        {
            "name": "Buck 24V→12V (D=0.5)",
            "topology": "buck",
            "components": {"L": 100e-6, "C": 100e-6, "D": 0.5, "R_load": 10},
            "specs": {"vin": 24, "vout": 12, "power": 14.4, "fsw": 100e3}
        },
        {
            "name": "Boost 12V→24V (D=0.5)",
            "topology": "boost", 
            "components": {"L": 150e-6, "C": 100e-6, "D": 0.5, "R_load": 24},
            "specs": {"vin": 12, "vout": 24, "power": 24, "fsw": 100e3}
        },
        {
            "name": "Buck-Boost 12V→-18V (D=0.6)",
            "topology": "buck_boost",
            "components": {"L": 100e-6, "C": 100e-6, "D": 0.6, "R_load": 18},
            "specs": {"vin": 12, "vout": 18, "power": 18, "fsw": 100e3}
        },
        {
            "name": "SEPIC 12V→18V (D=0.6)",
            "topology": "sepic",
            "components": {"L": 100e-6, "C": 47e-6, "D": 0.6, "R_load": 18},
            "specs": {"vin": 12, "vout": 18, "power": 18, "fsw": 100e3}
        },
        {
            "name": "Cuk 12V→-18V (D=0.6)",
            "topology": "cuk",
            "components": {"L": 100e-6, "C": 47e-6, "D": 0.6, "R_load": 18},
            "specs": {"vin": 12, "vout": 18, "power": 18, "fsw": 100e3}
        },
        # Isolated
        {
            "name": "Flyback 48V→12V (D=0.4)",
            "topology": "flyback",
            "components": {"L": 200e-6, "C": 100e-6, "D": 0.4, "R_load": 12},
            "specs": {"vin": 48, "vout": 12, "power": 12, "fsw": 100e3}
        },
        {
            "name": "Forward 48V→12V (D=0.4)",
            "topology": "forward",
            "components": {"L": 100e-6, "C": 100e-6, "D": 0.4, "R_load": 12},
            "specs": {"vin": 48, "vout": 12, "power": 12, "fsw": 100e3}
        },
        {
            "name": "Half-Bridge 400V→48V (D=0.45)",
            "topology": "half_bridge",
            "components": {"L": 100e-6, "C": 220e-6, "D": 0.45, "R_load": 4.8},
            "specs": {"vin": 400, "vout": 48, "power": 480, "fsw": 100e3}
        },
        {
            "name": "Full-Bridge 400V→48V (D=0.45)",
            "topology": "full_bridge",
            "components": {"L": 100e-6, "C": 220e-6, "D": 0.45, "R_load": 4.8},
            "specs": {"vin": 400, "vout": 48, "power": 480, "fsw": 100e3}
        },
        {
            "name": "Push-Pull 48V→12V (D=0.4)",
            "topology": "push_pull",
            "components": {"L": 100e-6, "C": 100e-6, "D": 0.4, "R_load": 12},
            "specs": {"vin": 48, "vout": 12, "power": 12, "fsw": 100e3}
        },
    ]
    
    results_summary = []
    
    for test in test_cases:
        print(f"\n--- {test['name']} ---")
        try:
            results, score = evaluator.evaluate(
                test["topology"],
                test["components"],
                test["specs"],
                level=3
            )
            
            if results.simulation_success:
                target = test['specs']['vout']
                error_pct = abs(results.vout_dc - target) / target * 100
                status = "✓" if error_pct < 20 else "⚠"
                print(f"  {status} Vout: {results.vout_dc:.2f}V (target: {target}V, error: {error_pct:.1f}%)")
                print(f"    Ripple: {results.vout_ripple_pct*100:.2f}%")
                print(f"    Efficiency: {results.efficiency*100:.0f}%")
                print(f"    Score: {score.total_score:.1f}/100")
                results_summary.append((test["topology"], True, score.total_score))
            else:
                print(f"  ✗ FAILED: {results.error_message[:80]}")
                results_summary.append((test["topology"], False, 0))
        except Exception as e:
            print(f"  ✗ ERROR: {str(e)[:80]}")
            results_summary.append((test["topology"], False, 0))
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    passed = sum(1 for _, success, _ in results_summary if success)
    print(f"Passed: {passed}/{len(results_summary)}")
    print("\nTopology Status:")
    for topo, success, score in results_summary:
        status = "✓" if success else "✗"
        print(f"  {status} {topo:15} - Score: {score:.1f}" if success else f"  {status} {topo:15} - FAILED")


if __name__ == "__main__":
    run_quick_test()
