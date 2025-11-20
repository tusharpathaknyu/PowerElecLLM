# 🚀 Start Here - Action Plan

## Phase 1: Environment Setup (30 minutes)

### Step 1: Set Up Conda Environment
```bash
cd ~/Desktop/PowerElecLLM
conda env create -f environment.yml
conda activate power_electronics
```

### Step 2: Verify Installation
```bash
# Test Python
python --version  # Should be 3.10+

# Test PySpice
python -c "import PySpice; print('✅ PySpice installed')"

# Test OpenAI
python -c "from openai import OpenAI; print('✅ OpenAI installed')"
```

### Step 3: Get API Key

You'll need an API key to use LLMs for circuit generation. Here are options:

#### Option A: OpenAI API (Recommended for GPT-4o)

1. **Sign up/Login**:
   - Go to https://platform.openai.com
   - Create an account or log in

2. **Get API Key**:
   - Navigate to https://platform.openai.com/api-keys
   - Click "Create new secret key"
   - Give it a name (e.g., "PowerElecLLM")
   - **Copy the key immediately** - you won't see it again!
   - Format: `sk-...` (starts with "sk-")

3. **Add Credits** (if needed):
   - Go to https://platform.openai.com/account/billing
   - Add payment method and credits
   - GPT-4o costs ~$2.50 per 1M input tokens, ~$10 per 1M output tokens

4. **Usage**:
   ```bash
   python src/power_run.py --api_key="sk-your-key-here" --task_id=1
   ```

#### Option B: DeepSeek API (Cheaper Alternative)

1. **Sign up**:
   - Go to https://platform.deepseek.com
   - Create an account

2. **Get API Key**:
   - Navigate to API keys section
   - Create a new API key
   - Copy the key

3. **Usage**:
   ```bash
   python src/power_run.py --api_key="your-deepseek-key" --model="deepseek-chat" --task_id=1
   ```

#### Option C: Local Models (Ollama - Free)

1. **Install Ollama**:
   ```bash
   brew install ollama  # macOS
   # Or download from https://ollama.ai
   ```

2. **Download a model**:
   ```bash
   ollama pull llama3.2  # or mistral, codellama, etc.
   ```

3. **Usage** (requires code modification):
   - Modify `power_run.py` to use Ollama's local API
   - No API key needed, runs locally

#### Security Best Practices

⚠️ **Never commit API keys to Git!**

1. **Use environment variables** (recommended):
   ```bash
   # Set in your shell
   export OPENAI_API_KEY="sk-your-key-here"
   
   # Then use in code
   python src/power_run.py --api_key="$OPENAI_API_KEY" --task_id=1
   ```

2. **Or use a `.env` file** (add to `.gitignore`):
   ```bash
   # Create .env file
   echo "OPENAI_API_KEY=sk-your-key-here" > .env
   
   # Load in Python (requires python-dotenv package)
   ```

3. **Check `.gitignore`**:
   - Make sure `.env` and `*.key` files are ignored
   - Your `.gitignore` already includes these ✅

#### Quick Test

Test your API key works:
```bash
python -c "from openai import OpenAI; client = OpenAI(api_key='YOUR_KEY'); print('✅ API key works!')"
```

#### Using Your API Key (After Setup)

Your API key has been set up as an environment variable! Here's how to use it:

**Option 1: Use environment variable directly** (Recommended):
```bash
# The key is already in your ~/.zshrc, so it loads automatically
# Just use it in commands:
python src/power_run.py --api_key="$OPENAI_API_KEY" --task_id=1
```

**Option 2: Use it in current session**:
```bash
# If you haven't restarted terminal, export it:
export OPENAI_API_KEY="sk-proj-..."  # (already done for you)

# Then use:
python src/power_run.py --api_key="$OPENAI_API_KEY" --task_id=1
```

**Option 3: Pass directly** (less secure, but works):
```bash
python src/power_run.py --api_key="sk-proj-..." --task_id=1
```

⚠️ **Important Security Notes**:
- Your API key is now saved in `~/.zshrc` (safe, local file)
- Never share your API key publicly
- If you ever need to revoke it, go to https://platform.openai.com/api-keys
- The key is NOT in your project files (safe from Git commits) ✅

---

## Phase 2: Study Reference (1-2 hours)

### Step 4: Understand AnalogCoder Structure
If you have access to AnalogCoder repository:
```bash
# Navigate to AnalogCoder (if available)
cd ~/path/to/AnalogCoder
# Study these files:
# - gpt_run.py (main execution flow)
# - prompt_template.md (how prompts are structured)
# - problem_check/ (validation logic)
```

**Key things to understand:**
- How LLM prompts are constructed
- How SPICE code is extracted from LLM responses
- How circuits are validated with PySpice
- How iterative refinement works

---

## Phase 3: Core Implementation (Week 1)

### Step 5: Create Problem Set Structure
Create `problem_set.tsv` or `problem_set.json` with power electronics problems:

```python
# Example structure:
problems = [
    {
        "task_id": 1,
        "topology": "Buck",
        "input_voltage": 12,
        "output_voltage": 5,
        "output_power": 50,
        "switching_freq": 500,  # kHz
        "efficiency_target": 90,
        "input_nodes": ["Vin", "GND"],
        "output_nodes": ["Vout", "GND"]
    }
]
```

### Step 6: Implement Template Loading
In `src/power_run.py`, add:
```python
def load_template(template_path):
    """Load prompt template from file"""
    with open(template_path, 'r') as f:
        return f.read()

def fill_template(template, problem_spec):
    """Fill template with problem specifications"""
    # Replace placeholders like [TASK], [INPUT_VOLTAGE], etc.
    pass
```

### Step 7: Implement LLM Call Function
```python
def call_llm(client, prompt, model="gpt-4o", temperature=0.5):
    """Call LLM API and get response"""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a power electronics design expert."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature
    )
    return response.choices[0].message.content
```

### Step 8: Implement Code Extraction
```python
def extract_spice_code(llm_response):
    """Extract PySpice code from LLM response"""
    # Look for code blocks with ```python or ```spice
    # Return the code as a string
    pass
```

---

## Phase 4: PySpice Validation (Week 1-2)

### Step 9: Create Test Bench Structure
Create `problem_check/buck_converter_check.py`:
```python
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *
import sys

def validate_buck_converter(circuit_code):
    """Validate a buck converter design"""
    # Execute the circuit code
    # Run simulation
    # Check output voltage, ripple, efficiency
    pass
```

### Step 10: Implement Circuit Execution
```python
def execute_circuit(circuit_code, output_dir):
    """Execute PySpice circuit code and save results"""
    # Create a temporary file with circuit code
    # Run PySpice simulation
    # Parse results
    # Return success/failure with metrics
    pass
```

---

## Phase 5: Iterative Refinement (Week 2)

### Step 11: Implement Feedback Loop
```python
def refine_design(client, initial_code, validation_results, template):
    """Refine design based on validation feedback"""
    # Create feedback prompt
    # Call LLM again with feedback
    # Return improved code
    pass
```

### Step 12: Add Retry Logic
```python
def generate_with_retry(client, problem, template, max_retries=3):
    """Generate circuit with retry on failure"""
    for attempt in range(max_retries):
        # Generate code
        # Validate
        # If success, return
        # If failure, refine and retry
    pass
```

---

## Phase 6: First Working Example (Week 2-3)

### Step 13: Create Simple Buck Converter Problem
Start with the simplest case:
- Input: 12V
- Output: 5V
- Power: 10W
- Frequency: 200kHz

### Step 14: Test End-to-End
```bash
python src/power_run.py \
    --task_id=1 \
    --api_key="YOUR_API_KEY" \
    --num_per_task=1 \
    --model="gpt-4o"
```

### Step 15: Debug and Iterate
- Fix any PySpice errors
- Improve prompt template
- Add better error handling
- Test with different specifications

---

## Phase 7: Expand Features (Week 3-4)

### Step 16: Add More Topologies
- Boost converter
- Buck-Boost converter
- Flyback converter

### Step 17: Add Power Analysis
- Efficiency calculation
- Ripple analysis
- Regulation metrics

### Step 18: Add GaN Device Models
- Proper GaN HEMT modeling
- Gate drive considerations
- Switching loss calculations

---

## Quick Reference Commands

### Development Workflow
```bash
# Activate environment
conda activate power_electronics

# Run main script
python src/power_run.py --task_id=1 --api_key="YOUR_KEY"

# Test PySpice directly
python -c "from PySpice.Spice.Netlist import Circuit; print('OK')"

# Check git status
git status
git add .
git commit -m "Description"
git push
```

### File Structure to Create
```
PowerElecLLM/
├── src/
│   ├── power_run.py          # Main script (implement)
│   ├── problem_set.py        # Problem definitions (create)
│   ├── validation.py         # Validation logic (create)
│   └── utils.py              # Helper functions (create)
├── problem_check/
│   ├── buck_check.py         # Buck converter validator (create)
│   └── boost_check.py        # Boost converter validator (create)
└── benchmarks/
    └── problem_set.tsv       # Problem dataset (create)
```

---

## Priority Order

**Must Do First:**
1. ✅ Environment setup
2. ✅ Template loading function
3. ✅ LLM API call function
4. ✅ Code extraction function
5. ✅ Simple PySpice execution test

**Then:**
6. ✅ Problem set structure
7. ✅ Basic validation
8. ✅ First working buck converter

**Later:**
9. Iterative refinement
10. Multiple topologies
11. Advanced analysis

---

## Tips

1. **Start Small**: Get a simple buck converter working first
2. **Test Frequently**: Test each function as you write it
3. **Use Print Statements**: Debug with print() before adding logging
4. **Commit Often**: Make small commits after each working feature
5. **Reference PySpice Docs**: https://pyspice.fabrice-salvaire.fr/

---

## Need Help?

- Check `docs/getting_started.md` for installation issues
- Review `templates/power_electronics_template.md` for prompt structure
- Look at PySpice examples online
- Test with simple circuits first before complex ones

Good luck! 🚀

