# PowerElecLLM

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Benchmarks](https://img.shields.io/badge/Benchmarks-500_Problems-orange.svg)](benchmarks/)

AI-powered power electronics circuit design using Large Language Models.

## 🚀 Features

- **PowerElecBench**: Comprehensive benchmark suite with 500 hand-crafted problems across 4 difficulty levels
- Automated generation of power converter designs (Buck, Boost, SEPIC, Ćuk, Flyback, Forward, Half-Bridge, Full-Bridge)
- 10 converter topologies with complete design equations
- PySpice/NgSpice simulation framework (98.3% test pass rate)
- Power analysis (efficiency, ripple, regulation, thermal)

## 📊 PowerElecBench Benchmark Suite

| Level | Name | Problems | Description |
|-------|------|----------|-------------|
| Level 1 | Basic | 150 | Fundamental DC-DC converter design |
| Level 2 | Intermediate | 150 | Constrained design with efficiency/thermal limits |
| Level 3 | Advanced | 100 | Multi-objective optimization |
| Level 4 | Expert | 100 | Cutting-edge integrated systems |
| **Total** | | **500** | |

### Supported Topologies
Buck, Boost, SEPIC, Ćuk, Inverting Buck-Boost, Quasi-Resonant Buck, Flyback, Forward, Half-Bridge, Full-Bridge

## 📋 Requirements

- Python ≥ 3.10
- PySpice ≥ 1.5
- NgSpice v45+

## 🛠️ Installation

```bash
git clone https://github.com/tusharpathaknyu/PowerElecLLM.git
cd PowerElecLLM
conda env create -f environment.yml
conda activate power_electronics
./setup_ngspice.sh  # Install NgSpice
```

## 🎯 Quick Start

```bash
# Run benchmark validation
python reference_tests/run_reference_tests.py

# Run specific benchmark
python scripts/run_benchmark.py --level 1 --problem 1
```

## 📚 Documentation

- [Getting Started](docs/getting_started.md)
- [Benchmark Details](benchmarks/README.md)
- [Flow Explanation](FLOW_EXPLANATION.md)

## 🎓 Inspiration

This project extends concepts from [AnalogCoder](https://github.com/anonyanalog/AnalogCoder) (AAAI'25) to power electronics design.

## 📊 Project Status

✅ **PowerElecBench 2.0 Complete** - 500 problems across 4 levels

- [x] 10 converter topology implementations
- [x] 500 benchmark problems with solutions
- [x] PySpice simulation framework
- [x] 98.3% reference test coverage (118/120)
- [ ] LLM evaluation pipeline
- [ ] Multi-model comparison

## 🤝 Contributing

Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) first.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details

## 👤 Author

Tushar Pathak - [GitHub](https://github.com/tusharpathaknyu)

## 🙏 Acknowledgments

- [AnalogCoder](https://github.com/anonyanalog/AnalogCoder) team for the foundational framework
- PySpice community
- Power electronics research community

