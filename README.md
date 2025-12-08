# PowerElecLLM

A benchmark and evaluation framework for testing Large Language Models on power electronics circuit design. Can LLMs design DC-DC converters that actually work?

**TL;DR:** We tested 4 LLMs on 650 power electronics problems using SPICE simulation as ground truth. Best result: Fine-tuned GPT-4o achieves 25% accuracy. LLaMA 3.3 70B scores 2.3%.

---

## Benchmark Results

| Model | Accuracy | Notes |
|-------|----------|-------|
| Fine-tuned GPT-4o | **25.0%** | +17% over base |
| Fine-tuned GPT-4o-mini | 22.5% | Cost-effective |
| GPT-4o (base) | 21.4% | Baseline |
| LLaMA 3.3 70B | 2.3% | Lacks domain knowledge |

**Pass criteria:** Simulated output voltage within 10% of target (ngspice verification)

### Key Findings

1. **Fine-tuning helps but doesn't solve the problem** - 17% relative improvement
2. **LLaMA fundamentally misunderstands power electronics** - confuses Vout with duty cycle
3. **Power electronics is hard for LLMs** - even the best model fails 3 out of 4 problems
4. **SPICE validation is essential** - without simulation, we'd never know these designs fail

---

## What This Project Does

1. **PowerElecBench** - 500 hand-crafted power electronics design problems across 4 difficulty levels
2. **SPICE Evaluator** - ngspice-based circuit simulation to verify LLM designs actually work
3. **Fine-tuning Pipeline** - Tools to create training datasets and fine-tune models
4. **Multi-model Evaluation** - Test GPT-4o, LLaMA, Gemini, Grok on the same problems

### Problem Example

```
Input: "Design a buck converter: 12V input to 5V output, 2A load"

Expected Output:
- Topology: Buck
- Duty Cycle: D = Vout/Vin = 0.417
- Inductor: L = 47uH
- Capacitor: C = 100uF

Validation: ngspice simulates the circuit and measures actual Vout
Pass if: |Vout_actual - 5V| / 5V < 10%
```

---

## Project Structure

```
PowerElecLLM/
├── benchmarks/
│   ├── level_1/          # 150 basic problems
│   ├── level_2/          # 150 intermediate problems
│   ├── level_3/          # 100 advanced problems
│   ├── level_4/          # 100 expert problems
│   ├── test_set_v2/      # 150 test problems
│   ├── finetune/         # Fine-tuning datasets (OpenAI, LLaMA formats)
│   └── results/          # Evaluation results
├── src/
│   ├── spice_evaluator.py    # SPICE simulation engine (10 topologies)
│   └── power_run.py          # Main runner
├── scripts/
│   ├── run_spice_eval_650.py # Run full evaluation
│   ├── create_finetune_dataset.py
│   └── ...
└── .env.example          # API key template
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- ngspice (circuit simulator)
- API keys for OpenAI (required), Google/xAI (optional)

### Installation

```bash
# Clone the repository
git clone https://github.com/tusharpathaknyu/PowerElecLLM.git
cd PowerElecLLM

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install ngspice
# macOS:
brew install ngspice
# Ubuntu:
sudo apt install ngspice

# Set up API keys
cp .env.example .env
# Edit .env and add your API keys
```

### Running Evaluations

```bash
# Evaluate GPT-4o on test set
python scripts/run_spice_eval_650.py --model gpt-4o --num 50

# Evaluate fine-tuned model
python scripts/run_spice_eval_650.py --model ft:gpt-4o:your-model-id --num 50
```

### Creating Fine-tuning Datasets

```bash
# Generate OpenAI-format JSONL for fine-tuning
python scripts/create_finetune_dataset.py
# Output: benchmarks/finetune/powerelec_finetune.jsonl

# Upload to OpenAI and fine-tune via their dashboard
```

---

## Supported Topologies

The SPICE evaluator supports 10 DC-DC converter topologies:

| Topology | Type | Transfer Function |
|----------|------|-------------------|
| Buck | Step-down | Vout = Vin * D |
| Boost | Step-up | Vout = Vin / (1-D) |
| Buck-Boost | Inverting | Vout = Vin * D / (1-D) |
| SEPIC | Non-inverting | Vout = Vin * D / (1-D) |
| Cuk | Inverting | Vout = Vin * D / (1-D) |
| Flyback | Isolated | Vout = Vin * N * D / (1-D) |
| Forward | Isolated | Vout = Vin * N * D |
| Half-Bridge | Isolated | Vout = Vin * N * D |
| Full-Bridge | Isolated | Vout = Vin * N * 2D |
| Push-Pull | Isolated | Vout = Vin * N * 2D |

---

## How SPICE Validation Works

1. **LLM generates design** - duty cycle, inductor, capacitor values
2. **Netlist generated** - Parameters plugged into topology-specific ngspice template
3. **Transient simulation** - ngspice runs 10ms simulation
4. **Output extraction** - Mean Vout calculated from steady-state waveform
5. **Pass/Fail** - Compare to target with 10% tolerance

```
Example ngspice netlist (Buck converter):
Vin input 0 DC 12
Vctrl ctrl 0 PULSE(0 5 0 1n 1n 2.08u 5u)  ; D=0.417, fsw=200kHz
S1 input sw ctrl 0 SWMOD
D1 0 sw DMOD
L1 sw vout 47u
C1 vout 0 100u
R1 vout 0 2.5
```

---

## Difficulty Levels

| Level | Accuracy | Description |
|-------|----------|-------------|
| Level 1 | 28.7% | Basic single-objective design |
| Level 2 | 31.6% | Constrained design (efficiency, thermal) |
| Level 3 | 22.2% | Multi-objective optimization |
| Level 4 | 12.0% | Complex real-world applications |

---

## API Configuration

Create a `.env` file (never commit this):

```bash
OPENAI_API_KEY=your_openai_key_here
GOOGLE_API_KEY=your_google_key_here    # Optional, for Gemini
XAI_API_KEY=your_xai_key_here          # Optional, for Grok
```

---

## Future Work

- Chain-of-thought prompting for step-by-step calculations
- Tool use (calculator, formula lookup)
- Retrieval-augmented generation with textbook formulas
- Larger fine-tuning datasets
- Waveform-based scoring (ripple, THD, settling time)

---

## Contributing

Contributions welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT License - see [LICENSE](LICENSE)

## Author

Tushar Pathak


## Acknowledgments

- Inspired by [AnalogCoder](https://github.com/anonyanalog/AnalogCoder) (AAAI'25)
- ngspice open-source circuit simulator

