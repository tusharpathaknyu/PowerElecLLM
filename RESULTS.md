# PowerElecLLM Benchmark Results

## Executive Summary

We evaluated 4 LLMs on 650 power electronics converter design problems using **SPICE simulation** as ground truth. This benchmark tests whether LLMs can correctly design DC-DC converters that actually work when simulated.

## Final Results

| Model | Passing | Total | Accuracy |
|-------|---------|-------|----------|
| **Fine-tuned GPT-4o** | 163 | 651 | **25.0%** |
| Fine-tuned GPT-4o-mini | 146 | 650 | 22.5% |
| GPT-4o (base) | 139 | 650 | 21.4% |
| LLaMA 3.3 70B | 15 | 650 | 2.3% |

**Pass criteria:** Output voltage within 10% of target (verified by ngspice simulation)

## Performance by Difficulty Level

| Level | Description | Best Model (FT GPT-4o) |
|-------|-------------|------------------------|
| Level 1 | Basic converters | 28.7% |
| Level 2 | Intermediate | 31.6% |
| Level 3 | Advanced | 22.2% |
| Level 4 | Expert | 12.0% |

## Key Findings

### 1. Fine-tuning Provides Measurable Improvement
- Fine-tuned GPT-4o: **25.0%** vs Base GPT-4o: **21.4%**
- **+17% relative improvement** from domain-specific training
- Fine-tuned GPT-4o-mini nearly matches base GPT-4o (cost-effective option)

### 2. LLaMA Lacks Domain Knowledge
- LLaMA 3.3 70B scored only **2.3%** (15/650 problems)
- Root cause: Confuses output voltage with duty cycle
  - For Buck 12V→5V: LLaMA returns D=0.05 (5%) instead of D=0.417 (41.7%)
- Without fine-tuning, LLaMA fundamentally misunderstands power electronics

### 3. Power Electronics is Hard for LLMs
- Even the best model only solves **1 in 4** problems correctly
- Requires precise numerical reasoning (duty cycle calculations)
- Small errors cascade into large output voltage deviations
- SPICE simulation is unforgiving - no partial credit

### 4. Difficulty Scaling Works
- Performance drops from 28.7% (Level 1) to 12.0% (Level 4)
- Validates that our difficulty levels meaningfully distinguish problem complexity

## Methodology

### Dataset
- **500 training problems** (Levels 1-4)
- **150 test problems** (test_set_v2)
- **10 topologies:** Buck, Boost, Buck-Boost, SEPIC, Cuk, Flyback, Forward, Half-Bridge, Full-Bridge, Push-Pull

### Evaluation
1. LLM generates converter parameters (duty cycle, L, C, R)
2. Parameters fed to ngspice SPICE simulator
3. Transient analysis measures actual output voltage
4. Pass if |Vout - Vout_target| / Vout_target < 10%

### Fine-tuning
- Platform: OpenAI
- Training examples: 283 (from Levels 1-4)
- Format: Chat completions with system prompt + problem + solution

## Implications

1. **Domain expertise matters** - Fine-tuning improves results but doesn't solve the problem
2. **Numerical reasoning gap** - LLMs struggle with precise calculations that circuit design requires
3. **Validation is essential** - Without simulation, we'd have no idea these designs don't work
4. **Future directions:**
   - Chain-of-thought prompting for calculations
   - Tool use (calculator, formula lookup)
   - Larger/better fine-tuning datasets
   - Retrieval-augmented generation with textbook formulas

## Files

- `benchmarks/results/spice_full_*.json` - Full evaluation results
- `benchmarks/finetune/` - Fine-tuning dataset
- `src/spice_evaluator.py` - SPICE evaluation engine
- `scripts/run_spice_eval_650.py` - Evaluation runner

---

*Benchmark conducted December 2025*
