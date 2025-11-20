# Getting Started with PowerElecLLM

## Installation

### Prerequisites
- Python ≥ 3.10
- Conda (recommended) or pip

### Setup

```bash
# Clone the repository
git clone https://github.com/tusharpathaknyu/PowerElecLLM.git
cd PowerElecLLM

# Create conda environment
conda env create -f environment.yml
conda activate power_electronics

# Verify installation
python -c "import PySpice; print('PySpice installed successfully')"
```

## Quick Start

### 1. Get an API Key

You'll need an OpenAI API key (or compatible LLM API):
- Sign up at https://platform.openai.com
- Get your API key from https://platform.openai.com/api-keys

### 2. Run Your First Generation

```bash
python src/power_run.py --task_id=1 --api_key="YOUR_API_KEY" --num_per_task=1
```

### 3. Check Results

Results will be in `[model_name]/p[task_id]/[iteration]/`

## Project Structure

```
PowerElecLLM/
├── src/              # Main source code
├── templates/        # Prompt templates
├── problem_check/    # Validation test benches
├── examples/        # Example circuits
├── docs/            # Documentation
└── benchmarks/      # Benchmark datasets
```

## Next Steps

- Read the [Career Roadmap](CAREER_ROADMAP.md) for strategic planning
- Follow the [Quick Start Action Plan](QUICK_START_ACTION_PLAN.md) for your first 2 weeks
- Check out [examples](../examples/) for sample circuits

## Troubleshooting

### PySpice Installation Issues
```bash
# Try installing from conda-forge
conda install -c conda-forge pyspice
```

### API Key Issues
- Make sure your API key is valid
- Check your API usage limits
- Verify the key format (starts with `sk-`)

## Getting Help

- Open an issue on GitHub
- Check existing issues
- Read the documentation

Happy designing! 🚀

