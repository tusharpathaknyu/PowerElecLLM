# PowerElecLLM

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

AI-powered power electronics circuit design using Large Language Models.

## 🚀 Features

- Automated generation of power converter designs (Buck, Boost, Flyback, PFC)
- GaN device support with proper modeling
- Power analysis (efficiency, ripple, regulation)
- SPICE code generation for simulation
- Iterative refinement with automated validation

## 📋 Requirements

- Python ≥ 3.10
- PySpice ≥ 1.5
- OpenAI API key (or compatible LLM API)

## 🛠️ Installation

```bash
git clone https://github.com/tusharpathaknyu/PowerElecLLM.git
cd PowerElecLLM
conda env create -f environment.yml
conda activate power_electronics
```

## 🎯 Quick Start

```bash
python src/power_run.py --task_id=1 --api_key="YOUR_KEY" --num_per_task=1
```

## 📚 Documentation

- [Getting Started](docs/getting_started.md)
- [Career Roadmap](docs/CAREER_ROADMAP.md)
- [Examples](examples/)

## 🎓 Inspiration

This project extends concepts from [AnalogCoder](https://github.com/anonyanalog/AnalogCoder) (AAAI'25) to power electronics design.

## 🔑 Key Differences from AnalogCoder

- **Focus**: Power electronics (vs analog IC circuits)
- **Components**: Support for inductors, transformers, GaN devices
- **Analysis**: Power-specific metrics (efficiency, ripple, regulation)
- **Topologies**: Power converter topologies and design equations

## 📊 Project Status

🚧 **Under Active Development** 🚧

- [x] Project setup and structure
- [ ] Inductor and transformer support
- [ ] Power electronics prompt templates
- [ ] Buck converter generation
- [ ] Multiple topology support
- [ ] GaN device modeling
- [ ] Power analysis framework
- [ ] 65W charger system

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

