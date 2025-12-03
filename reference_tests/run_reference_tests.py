"""Reference regression tests for PowerElecLLM buck/boost converters.

This script programmatically instantiates parameterized buck and boost converters
using PySpice, runs transient simulations, and prints summary metrics so we can
be confident the template + validation logic generalize beyond a single spec.

Test coverage:
- Buck converters: two specs (12V->5V@10W, 24V->12V@50W)
- Boost converters: two specs (5V->12V@20W, 9V->24V@40W)

Each test prints:
- Target vs measured output voltage (mean over final 20% window)
- Absolute/t% error and ripple
- Peak switching node voltage (checks topology correctness)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *


PLOT_DIR = Path(__file__).parent / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class ConverterResult:
    name: str
    vin: float
    vout_target: float
    vout_actual: float
    error_pct: float
    ripple: float
    vsw_max: float
    vsw_min: float
    duty_cycle: float
    sim_time_ms: float
    plot_path: Path


def _simulate_circuit(
    circuit: Circuit,
    f_sw: float,
    duration_ms: float,
    name: str,
    target_vout: float,
    duty_cycle: float,
    vin: float,
) -> ConverterResult:
    period = 1.0 / f_sw
    step_time = period / 200
    end_time = duration_ms / 1000

    simulator = circuit.simulator(temperature=25, nominal_temperature=25)
    analysis = simulator.transient(step_time=step_time@u_s, end_time=end_time@u_s)

    time_s = np.array(analysis.time)
    time_ms = time_s * 1000
    vout = np.array(analysis['Vout'])
    vsw = np.array(analysis['Vsw'])
    vgate = np.array(analysis['Vgate'])

    steady_idx = int(len(vout) * 0.8)
    vout_final = float(np.mean(vout[steady_idx:]))
    ripple = float(np.max(vout[steady_idx:]) - np.min(vout[steady_idx:]))

    plot_path = PLOT_DIR / f"{name}.png"
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    ax1.plot(time_ms, vout, label='Vout')
    ax1.axhline(target_vout, color='r', linestyle='--', label='Target')
    ax1.set_ylabel('Vout (V)')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')

    ax2.plot(time_ms, vsw, color='g', label='Vsw')
    ax2.set_ylabel('Vsw (V)')
    ax2.grid(True, alpha=0.3)

    ax3.plot(time_ms, vgate, color='purple', label='Vgate')
    ax3.set_ylabel('Vgate (V)')
    ax3.set_xlabel('Time (ms)')
    ax3.grid(True, alpha=0.3)

    fig.suptitle(f"{name} simulation")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    return ConverterResult(
        name=name,
        vin=vin,
        vout_target=target_vout,
        vout_actual=vout_final,
        error_pct=(vout_final - target_vout) / target_vout * 100,
        ripple=ripple,
        vsw_max=float(np.max(vsw)),
        vsw_min=float(np.min(vsw)),
        duty_cycle=duty_cycle,
        sim_time_ms=duration_ms,
        plot_path=plot_path,
    )


def _design_buck_components(
    vin: float,
    vout: float,
    power: float,
    f_sw: float,
    i_ripple_frac: float = 0.25,
    v_ripple_frac: float = 0.01,
) -> Tuple[float, float, float, float, float]:
    i_load = power / vout
    delta_i = max(i_ripple_frac * i_load, 0.1)
    duty = vout / vin
    l_value = (vin - vout) * duty / (delta_i * f_sw)
    delta_v = max(v_ripple_frac * vout, 0.01)
    c_value = i_load * duty / (delta_v * f_sw)
    r_load = (vout ** 2) / power
    r_switch = 0.05
    r_inductor = 0.02
    return l_value, c_value, r_load, r_switch, r_inductor


def _build_buck_converter(
    name: str,
    vin: float,
    vout: float,
    power: float,
    f_sw: float,
    duration_ms: float,
) -> Tuple[Circuit, float, float, float]:
    l_value, c_value, r_load, r_switch, r_inductor = _design_buck_components(vin, vout, power, f_sw)
    i_load = power / vout
    v_diode = 0.4
    loss = v_diode + i_load * (r_switch + r_inductor)
    duty = min(max((vout + loss) / vin, 0.05), 0.95)
    period = 1.0 / f_sw

    circuit = Circuit(name)
    circuit.V('in', 'Vin', circuit.gnd, vin@u_V)
    circuit.VCS('S1', 'Vin', 'Vsw', 'Vgate', circuit.gnd, model='SMOD')
    circuit.model('SMOD', 'SW', Ron=r_switch@u_Ω, Roff=1@u_MΩ, Vt=2.5, Vh=0.5)
    circuit.model('DMOD', 'D', **{'is': 1e-9}, Rs=0.02, N=1.5)
    circuit.D('D1', circuit.gnd, 'Vsw', model='DMOD')
    circuit.L('L1', 'Vsw', 'Vout', l_value@u_H)
    circuit.C('C1', 'Vout', circuit.gnd, c_value@u_F, initial_condition=0@u_V)
    circuit.R('load', 'Vout', circuit.gnd, r_load@u_Ω)

    pulse_width = duty * period
    circuit.PulseVoltageSource(
        'gate', 'Vgate', circuit.gnd,
        initial_value=0@u_V,
        pulsed_value=5@u_V,
        pulse_width=pulse_width@u_s,
        period=period@u_s,
        delay_time=0@u_ns,
        rise_time=20@u_ns,
        fall_time=20@u_ns,
    )

    return circuit, duty, vin, duration_ms


def _design_boost_components(
    vin: float,
    vout: float,
    power: float,
    f_sw: float,
    i_ripple_frac: float = 0.3,
    v_ripple_frac: float = 0.02,
) -> Tuple[float, float, float, float, float]:
    i_out = power / vout
    duty = 1 - (vin / vout)
    i_in = i_out / (1 - duty)
    delta_i = max(i_ripple_frac * i_in, 0.1)
    l_value = vin * duty / (delta_i * f_sw)
    delta_v = max(v_ripple_frac * vout, 0.01)
    c_value = i_out * duty / (delta_v * f_sw)
    r_load = (vout ** 2) / power
    r_switch = 0.07
    r_inductor = 0.03
    return l_value, c_value, r_load, r_switch, r_inductor


def _build_boost_converter(
    name: str,
    vin: float,
    vout: float,
    power: float,
    f_sw: float,
    duration_ms: float,
) -> Tuple[Circuit, float, float, float]:
    l_value, c_value, r_load, r_switch, r_inductor = _design_boost_components(vin, vout, power, f_sw)
    i_out = power / vout
    v_diode = 0.5
    loss = v_diode + i_out * (r_switch + r_inductor)
    effective_vin = vin - loss
    duty = min(max(1 - (effective_vin / vout), 0.05), 0.95)
    period = 1.0 / f_sw

    circuit = Circuit(name)
    circuit.V('in', 'Vin', circuit.gnd, vin@u_V)
    circuit.L('L1', 'Vin', 'Vsw', l_value@u_H)
    circuit.VCS('S1', 'Vsw', circuit.gnd, 'Vgate', circuit.gnd, model='SMOD')
    circuit.model('SMOD', 'SW', Ron=r_switch@u_Ω, Roff=1@u_MΩ, Vt=2.5, Vh=0.5)
    circuit.model('DMOD', 'D', **{'is': 1e-9}, Rs=0.05, N=1.5)
    circuit.D('D1', 'Vsw', 'Vout', model='DMOD')
    circuit.C('C1', 'Vout', circuit.gnd, c_value@u_F, initial_condition=0@u_V)
    circuit.R('load', 'Vout', circuit.gnd, r_load@u_Ω)

    pulse_width = duty * period
    circuit.PulseVoltageSource(
        'gate', 'Vgate', circuit.gnd,
        initial_value=0@u_V,
        pulsed_value=5@u_V,
        pulse_width=pulse_width@u_s,
        period=period@u_s,
        delay_time=0@u_ns,
        rise_time=20@u_ns,
        fall_time=20@u_ns,
    )

    return circuit, duty, vin, duration_ms


def run_reference_tests() -> List[ConverterResult]:
    cases: List[ConverterResult] = []

    buck_specs = [
        ("buck_12v_to_5v", 12.0, 5.0, 10.0, 200e3, 8.0),
        ("buck_24v_to_12v", 24.0, 12.0, 50.0, 350e3, 8.0),
        ("buck_48v_to_5v", 48.0, 5.0, 40.0, 120e3, 12.0),
        ("buck_15v_to_3v3", 15.0, 3.3, 25.0, 250e3, 10.0),
    ]
    for name, vin, vout, power, f_sw, duration in buck_specs:
        circuit, duty, vin_ref, duration_ms = _build_buck_converter(name, vin, vout, power, f_sw, duration)
        result = _simulate_circuit(circuit, f_sw, duration_ms, name, vout, duty, vin_ref)
        cases.append(result)

    boost_specs = [
        ("boost_5v_to_12v", 5.0, 12.0, 20.0, 200e3, 12.0),
        ("boost_9v_to_24v", 9.0, 24.0, 40.0, 150e3, 15.0),
        ("boost_3v3_to_15v", 3.3, 15.0, 15.0, 180e3, 18.0),
        ("boost_12v_to_48v", 12.0, 48.0, 60.0, 100e3, 20.0),
    ]
    for name, vin, vout, power, f_sw, duration in boost_specs:
        circuit, duty, vin_ref, duration_ms = _build_boost_converter(name, vin, vout, power, f_sw, duration)
        result = _simulate_circuit(circuit, f_sw, duration_ms, name, vout, duty, vin_ref)
        cases.append(result)

    return cases


def print_summary(results: List[ConverterResult], tolerance_pct: float = 5.0) -> None:
    print("\n=== Reference Test Summary ===")
    for res in results:
        status = "PASS" if abs(res.error_pct) <= tolerance_pct else "FAIL"
        print(
            f"{res.name:18s} | Vout target {res.vout_target:5.2f} V | "
            f"measured {res.vout_actual:5.2f} V | error {res.error_pct:+5.2f}% | "
            f"ripple {res.ripple*1000:6.1f} mV | {status}"
        )
        print(f"  Plot: {res.plot_path}")


def all_within_tolerance(results: List[ConverterResult], tolerance_pct: float = 5.0) -> bool:
    return all(abs(res.error_pct) <= tolerance_pct for res in results)


if __name__ == "__main__":
    results = run_reference_tests()
    print_summary(results)
