# 🚀 Quick Start Guide

## You're Ready to Generate Circuits!

### Step 1: Activate Environment
```bash
conda activate power_electronics
```

### Step 2: Run Your First Generation

**Option A: Using environment variable (recommended)**
```bash
python src/power_run.py --task_id=1 --num_per_task=1
```

**Option B: Pass API key directly**
```bash
python src/power_run.py --task_id=1 --api_key="$OPENAI_API_KEY" --num_per_task=1
```

### Step 3: Check Results

Results will be saved in:
```
gpt_4o/task_1/iteration_1/circuit.py
```

### Available Tasks

- **Task 1**: Buck converter (12V → 5V, 10W)
- **Task 2**: Buck converter (24V → 12V, 50W)  
- **Task 3**: Boost converter (5V → 12V, 20W)

### Command Options

```bash
python src/power_run.py \
    --task_id=1 \              # Which problem to solve
    --num_per_task=1 \        # How many iterations
    --model="gpt-4o" \         # LLM model (default: gpt-4o)
    --temperature=0.5 \        # Creativity (0.0-1.0)
    --api_key="sk-..."         # Optional if OPENAI_API_KEY is set
```

### What Happens

1. ✅ Loads problem specification
2. ✅ Loads and fills prompt template
3. ✅ Calls LLM to generate circuit code
4. ✅ Extracts PySpice code from response
5. ✅ Saves code to file

### Next Steps

After generation, you'll need to:
1. **Validate the code** - Run PySpice simulation
2. **Check results** - Verify output voltage, ripple, etc.
3. **Refine if needed** - Iterate with feedback

See `START_HERE.md` for full development roadmap!

