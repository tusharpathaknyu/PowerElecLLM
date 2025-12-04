# PowerElecBench: Power Electronics Benchmark Dataset

A comprehensive benchmark for evaluating LLM capabilities in power electronics design.

## Overview

PowerElecBench contains **600 problems** with ground truth solutions across 4 difficulty levels:

| Version | Problems | Description |
|---------|----------|-------------|
| v1 (Hand-crafted) | 100 | Detailed problems with design rationale |
| v2 (Generated) | 500 | Parametrically generated with solutions |
| **Total** | **600** | |

## Problem Levels

### Level 1: Basic Converter Design (180 problems)
Single-stage converters with fixed specifications.
```json
{
  "prompt": "Design a buck converter: 12V to 5V, 3A for USB power delivery",
  "topology": "buck",
  "specifications": {"vin": 12, "vout": 5, "iout": 3}
}
```

### Level 2: Constrained Design (240 problems)
Converters with additional design constraints.
```json
{
  "prompt": "Design a boost converter with ripple < 50mV and efficiency > 92%",
  "constraints": {"ripple_mv": 50, "efficiency_min": 0.92}
}
```

### Level 3: Multi-Stage Systems (120 problems)
Complex systems requiring multiple converter stages.
```json
{
  "prompt": "Design a universal input (85-265VAC) to 19V laptop adapter",
  "stages": ["pfc_boost", "llc_resonant"]
}
```

### Level 4: Control Design (60 problems)
Compensator and control system design.
```json
{
  "prompt": "Design Type III compensator for 12V→1.2V VRM with 50kHz crossover",
  "control_type": "voltage_mode"
}
```

## Topologies Covered

- **Non-isolated**: Buck, Boost, SEPIC, Ćuk, Inverting Buck-Boost, QR-Buck
- **Isolated**: Flyback, Forward, Half-Bridge, Full-Bridge, LLC Resonant, DAB

## File Structure

```
benchmarks/
├── powerelec_bench.json          # v1 manifest (100 problems)
├── generate_benchmark.py         # Generator script
├── level_1/                      # v1 Level 1 problems + solutions
├── level_2/                      # v1 Level 2 problems + solutions
├── level_3/                      # v1 Level 3 problems + solutions
├── level_4/                      # v1 Level 4 problems + solutions
└── generated/                    # v2 generated problems
    ├── powerelec_bench_500.json  # v2 manifest
    ├── level_1/                  # 150 problems
    ├── level_2/                  # 200 problems
    ├── level_3/                  # 100 problems
    └── level_4/                  # 50 problems
```

## Usage

### Load the Benchmark

```python
import json

# Load v2 manifest
with open('benchmarks/generated/powerelec_bench_500.json') as f:
    manifest = json.load(f)

# Load Level 1 problems
with open('benchmarks/generated/level_1/problems_001_050.json') as f:
    level1 = json.load(f)
    
for problem in level1['problems']:
    print(f"{problem['id']}: {problem['prompt']}")
```

### Generate More Problems

```bash
python benchmarks/generate_benchmark.py
```

Customize the counts in the script:
```python
generator.generate_all(
    level_1_count=150,
    level_2_count=200,
    level_3_count=100,
    level_4_count=50
)
```

## Evaluation Metrics

| Metric | Threshold | Description |
|--------|-----------|-------------|
| Voltage Accuracy | < 5% error | Output voltage vs. specification |
| Component Selection | Within 2x | Inductor, capacitor values |
| Efficiency Estimate | Within 5% | Predicted vs. reference |
| Constraint Satisfaction | 100% | All constraints met |

## Solution Format

Each problem includes a ground truth solution:

```json
{
  "solution": {
    "topology": "buck",
    "duty_cycle": 0.417,
    "components": {
      "L": {"value": 33e-6, "unit": "H", "rating": "5A"},
      "C_out": {"value": 47e-6, "unit": "F", "type": "ceramic"}
    },
    "design_equations": {
      "duty_cycle": "D = Vout/Vin = 5/12 = 0.417",
      "inductor": "L = Vout×(1-D)/(fsw×ΔIL) = 33µH"
    },
    "expected_results": {
      "vout_actual": 5.0,
      "ripple_mv": 25,
      "efficiency_est": 0.94
    }
  }
}
```

## Application Domains

- **Consumer**: USB-C PD, laptop adapters, LED drivers
- **Automotive**: EV chargers, 48V mild hybrid, battery management
- **Industrial**: Motor drives, welding power, process control
- **Telecom**: -48V systems, rectifiers, backup power
- **Data Center**: VRMs, server PSUs, UPS systems
- **Renewable**: Solar MPPT, battery storage, grid-tie inverters

## Citation

If you use PowerElecBench in your research, please cite:
```bibtex
@misc{powerelecbench2024,
  title={PowerElecBench: A Benchmark for LLM-Based Power Electronics Design},
  author={Pathak, Tushar},
  year={2024},
  url={https://github.com/tusharpathaknyu/PowerElecLLM}
}
```

## License

MIT License - see [LICENSE](../LICENSE) for details.
