# 🔄 System Flow Explanation

## Complete Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    USER RUNS COMMAND                             │
│  python src/power_run.py --task_id=1 --num_per_task=1            │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: Parse Arguments & Initialize                            │
│  ─────────────────────────────────────────────────────────────── │
│  • --task_id=1 → args.task_id = 1                                 │
│  • --num_per_task=1 → args.num_per_task = 1                      │
│  • Get API key from env var or args                               │
│  • Initialize OpenAI client                                       │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2: Load Problem Specification                              │
│  ─────────────────────────────────────────────────────────────── │
│  File: benchmarks/problem_set.json                                │
│                                                                   │
│  load_problem_set() → Reads JSON file                            │
│  get_problem_by_id(problems, 1) → Finds task_id=1                │
│                                                                   │
│  Returns:                                                         │
│  {                                                                │
│    "task_id": 1,                                                  │
│    "topology": "Buck",                                            │
│    "input_voltage": 12,                                           │
│    "output_voltage": 5,                                           │
│    "output_power": 10,                                            │
│    "switching_freq": 200,                                         │
│    "efficiency_target": 85,                                       │
│    "input_nodes": ["Vin", "GND"],                                 │
│    "output_nodes": ["Vout", "GND"]                                │
│  }                                                                │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 3: Load & Fill Template                                    │
│  ─────────────────────────────────────────────────────────────── │
│  File: templates/power_electronics_template.md                   │
│                                                                   │
│  load_template() → Reads template file                           │
│  fill_template(template, problem) → Replaces placeholders       │
│                                                                   │
│  BEFORE (Template):                                              │
│  "Design [TASK].                                                 │
│   Input voltage: [INPUT_VOLTAGE]V                                │
│   Output voltage: [OUTPUT_VOLTAGE]V"                             │
│                                                                   │
│  AFTER (Filled Prompt):                                          │
│  "Design Buck converter.                                        │
│   Input voltage: 12V                                             │
│   Output voltage: 5V                                             │
│   Output power: 10W                                               │
│   Switching frequency: 200kHz                                    │
│   Efficiency target: 85%"                                        │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 4: Call LLM API                                            │
│  ─────────────────────────────────────────────────────────────── │
│  call_llm(client, prompt, model="gpt-4o")                       │
│                                                                   │
│  API Call:                                                        │
│  {                                                                │
│    "model": "gpt-4o",                                             │
│    "messages": [                                                  │
│      {                                                            │
│        "role": "system",                                          │
│        "content": "You are a power electronics expert..."        │
│      },                                                           │
│      {                                                            │
│        "role": "user",                                            │
│        "content": "<filled prompt from step 3>"                  │
│      }                                                            │
│    ],                                                             │
│    "temperature": 0.5                                             │
│  }                                                                │
│                                                                   │
│  Returns: LLM response (text with explanation + code)            │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 5: Extract PySpice Code                                   │
│  ─────────────────────────────────────────────────────────────── │
│  extract_spice_code(llm_response)                               │
│                                                                   │
│  LLM Response (example):                                         │
│  "Here's the buck converter design...                           │
│                                                                   │
│   ```python                                                      │
│   from PySpice.Spice.Netlist import Circuit                     │
│   from PySpice.Unit import *                                     │
│   circuit = Circuit('Buck Converter 12V to 5V')                 │
│   ...                                                            │
│   ```"                                                           │
│                                                                   │
│  Extraction Process:                                             │
│  1. Look for ```python code blocks                               │
│  2. Extract content between ```                                  │
│  3. Clean up whitespace                                           │
│                                                                   │
│  Returns: Pure Python code string                                │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 6: Save Generated Code                                     │
│  ─────────────────────────────────────────────────────────────── │
│  save_generated_code(code, output_dir, task_id, iteration)      │
│                                                                   │
│  Creates directory structure:                                    │
│  gpt_4o/                                                         │
│    └── task_1/                                                   │
│        └── iteration_1/                                          │
│            └── circuit.py  ← Saved here                          │
│                                                                   │
│  Writes code to file                                             │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 7: (TODO) Validate Circuit                                │
│  ─────────────────────────────────────────────────────────────── │
│  This is where you'll add:                                      │
│                                                                   │
│  1. Run PySpice simulation                                       │
│  2. Check output voltage (should be ~5V)                        │
│  3. Measure ripple                                               │
│  4. Calculate efficiency                                          │
│  5. Return pass/fail + metrics                                    │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 8: (TODO) Iterative Refinement                             │
│  ─────────────────────────────────────────────────────────────── │
│  If validation fails:                                            │
│                                                                   │
│  1. Extract error messages                                       │
│  2. Create feedback prompt:                                      │
│     "Previous attempt failed: [errors]                         │
│      Fix these issues: [specific problems]"                      │
│  3. Call LLM again with feedback                                 │
│  4. Retry up to --num_of_retry times                             │
└─────────────────────────────────────────────────────────────────┘
```

## Detailed Function Flow

### Example: Running `python src/power_run.py --task_id=1`

#### 1. **main() function starts** (line 168)
```python
main()
  ↓
```

#### 2. **Load Problem** (lines 176-184)
```python
problems = load_problem_set()
  → Reads: benchmarks/problem_set.json
  → Returns: [{"task_id": 1, ...}, {"task_id": 2, ...}, ...]

problem = get_problem_by_id(problems, 1)
  → Searches for task_id=1
  → Returns: {"task_id": 1, "topology": "Buck", "input_voltage": 12, ...}
```

#### 3. **Load & Fill Template** (lines 187-193)
```python
template = load_template()
  → Reads: templates/power_electronics_template.md
  → Returns: "You aim to design... [TASK] ... [INPUT_VOLTAGE]V ..."

prompt = fill_template(template, problem)
  → Replaces [TASK] with "Buck converter"
  → Replaces [INPUT_VOLTAGE] with "12"
  → Replaces [OUTPUT_VOLTAGE] with "5"
  → ... (all placeholders)
  → Returns: Complete prompt ready for LLM
```

#### 4. **Generate Loop** (lines 198-226)
```python
for iteration in range(1, 2):  # num_per_task=1
  ↓
  llm_response = call_llm(client, prompt, "gpt-4o")
    → Makes API call to OpenAI
    → Returns: Long text response with explanation + code
  ↓
  code = extract_spice_code(llm_response)
    → Searches for ```python blocks
    → Extracts code between ```
    → Returns: Clean Python code string
  ↓
  code_file = save_generated_code(code, "gpt_4o", 1, 1)
    → Creates: gpt_4o/task_1/iteration_1/circuit.py
    → Writes code to file
    → Returns: Path to saved file
```

## Data Transformation Example

### Input → Output Transformation

**INPUT (Problem Spec):**
```json
{
  "task_id": 1,
  "input_voltage": 12,
  "output_voltage": 5,
  "output_power": 10
}
```

**PROCESSING:**
```
Problem → Template → Prompt → LLM → Response → Code Extraction → File
```

**OUTPUT (Generated Code):**
```python
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *

circuit = Circuit('Buck Converter 12V to 5V')
circuit.V('in', 'Vin', circuit.gnd, 12@u_V)
circuit.L('L1', 'Vsw', 'Vout', 24.3@u_uH)
circuit.C('C1', 'Vout', circuit.gnd, 100@u_uF)
# ... rest of circuit
```

## Current vs. Complete Flow

### ✅ What Works Now:
1. Problem loading ✓
2. Template filling ✓
3. LLM API calls ✓
4. Code extraction ✓
5. File saving ✓

### ❌ What's Missing (TODOs):
1. **Validation** (line 225)
   - Run PySpice simulation
   - Check if circuit works
   - Extract metrics

2. **Iterative Refinement** (line 226)
   - If validation fails, create feedback
   - Retry with improved prompt
   - Loop until success or max retries

## How to Add Validation

### Step 1: Create validation function
```python
def validate_circuit(code_file, expected_vout=5.0):
    """Run simulation and check results"""
    # Execute the circuit code
    # Run PySpice simulation
    # Extract Vout from results
    # Check if Vout ≈ expected_vout
    # Return: (passed: bool, metrics: dict, errors: list)
    pass
```

### Step 2: Integrate into main loop
```python
# After saving code (line 220):
result = validate_circuit(code_file, problem['output_voltage'])
if not result.passed:
    # Create feedback prompt
    feedback = create_feedback_prompt(code, result.errors)
    # Retry with feedback
    llm_response = call_llm(client, feedback, ...)
```

## Key Files & Their Roles

| File | Purpose | Used By |
|------|---------|---------|
| `benchmarks/problem_set.json` | Problem specifications | `load_problem_set()` |
| `templates/power_electronics_template.md` | Prompt template | `load_template()` |
| `src/power_run.py` | Main execution script | User runs this |
| `problem_check/buck_check.py` | Validation logic | (To be integrated) |
| `gpt_4o/task_X/iteration_Y/circuit.py` | Generated circuits | (Output) |

## Understanding the Code Structure

### Function Dependencies:
```
main()
  ├── load_problem_set()
  │     └── Reads: benchmarks/problem_set.json
  ├── get_problem_by_id()
  ├── load_template()
  │     └── Reads: templates/power_electronics_template.md
  ├── fill_template()
  ├── call_llm()  [External API call]
  ├── extract_spice_code()
  └── save_generated_code()
        └── Writes: gpt_4o/task_X/iteration_Y/circuit.py
```

### Data Flow Types:
1. **File I/O**: Reading JSON, templates; Writing generated code
2. **String Processing**: Template filling, code extraction (regex)
3. **API Calls**: LLM requests/responses
4. **File System**: Creating directories, saving files

This is the complete flow! Each step transforms data until you get executable PySpice code.


