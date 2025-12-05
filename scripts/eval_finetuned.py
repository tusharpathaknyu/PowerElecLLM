#!/usr/bin/env python3
"""
Evaluate fine-tuned GPT-4o-mini on PowerElecBench with progress saving
"""
import json
import os
import time
from pathlib import Path
from openai import OpenAI

# Config
MODEL = "ft:gpt-4o-mini-2024-07-18:personal:powerelec:CjJntxcS"
BENCHMARK_DIR = Path(__file__).parent.parent / "benchmarks"
RESULTS_FILE = BENCHMARK_DIR / "results" / "finetuned_4omini_results.json"

SYSTEM_PROMPT = """You are an expert power electronics engineer. Design circuits with optimal component values.

Return your answer in this exact JSON format:
{
    "topology": "buck|boost|buck_boost|flyback|forward|full_bridge|half_bridge|push_pull",
    "vout": <calculated output voltage as number>,
    "components": {
        "L": <inductance in H>,
        "C": <capacitance in F>,
        "R_load": <load resistance in Ω>,
        "D": <duty cycle 0-1>,
        "fsw": <switching frequency in Hz>,
        ... other relevant components
    },
    "explanation": "Brief explanation of design choices"
}"""

def load_problems():
    """Load all benchmark problems"""
    all_problems = []
    for level in [1, 2, 3, 4]:
        level_dir = BENCHMARK_DIR / f"level_{level}"
        for f in sorted(level_dir.glob("problems_*.json")):
            with open(f) as fp:
                data = json.load(fp)
                for p in data.get("problems", []):
                    p["level"] = level
                    all_problems.append(p)
    return all_problems

def evaluate_problem(client, problem):
    """Evaluate a single problem"""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": problem["prompt"]}
            ],
            max_tokens=2000,
            timeout=60
        )
        
        text = response.choices[0].message.content
        
        # Parse response
        try:
            # Find JSON in response
            import re
            json_match = re.search(r'\{[^{}]*"vout"[^{}]*\}', text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                llm_vout = float(result.get("vout", 0))
            else:
                # Try to find vout directly
                vout_match = re.search(r'"vout"\s*:\s*([\d.]+)', text)
                if vout_match:
                    llm_vout = float(vout_match.group(1))
                else:
                    llm_vout = None
        except:
            llm_vout = None
        
        # Calculate error
        gt_vout = problem.get("specs", {}).get("vout")
        if gt_vout and llm_vout:
            error = abs(llm_vout - gt_vout) / gt_vout * 100
            success = error < 5  # Within 5%
        else:
            error = None
            success = False
            
        return {
            "id": problem.get("id", "unknown"),
            "level": problem["level"],
            "gt_vout": gt_vout,
            "llm_vout": llm_vout,
            "error_pct": error,
            "success": success,
            "tokens_in": response.usage.prompt_tokens,
            "tokens_out": response.usage.completion_tokens
        }
        
    except Exception as e:
        return {
            "id": problem.get("id", "unknown"),
            "level": problem["level"],
            "gt_vout": problem.get("specs", {}).get("vout"),
            "llm_vout": None,
            "error_pct": None,
            "success": False,
            "error": str(e)
        }

def main():
    print(f"🔬 Evaluating fine-tuned GPT-4o-mini: {MODEL}")
    
    # Load existing results if any
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            results = json.load(f)
        completed_ids = {r["id"] for r in results}
        print(f"📁 Resuming from {len(results)} completed evaluations")
    else:
        results = []
        completed_ids = set()
        RESULTS_FILE.parent.mkdir(exist_ok=True)
    
    client = OpenAI()
    problems = load_problems()
    print(f"📊 Total problems: {len(problems)}")
    
    # Filter to uncompleted problems
    remaining = [p for p in problems if p.get("id") not in completed_ids]
    print(f"📝 Remaining: {len(remaining)}")
    
    total_tokens = 0
    for i, problem in enumerate(remaining):
        result = evaluate_problem(client, problem)
        results.append(result)
        
        status = "✅" if result["success"] else "❌"
        vout_str = f"Vout={result['llm_vout']}" if result['llm_vout'] else "parse failed"
        err_str = f"err={result['error_pct']:.1f}%" if result['error_pct'] is not None else ""
        
        print(f"[{i+1}/{len(remaining)}] L{result['level']}_{result['id']}: {status} {vout_str} {err_str}")
        
        # Track tokens
        if "tokens_in" in result:
            total_tokens += result.get("tokens_in", 0) + result.get("tokens_out", 0)
        
        # Save progress every 10 problems
        if (i + 1) % 10 == 0:
            with open(RESULTS_FILE, "w") as f:
                json.dump(results, f, indent=2)
            print(f"   💾 Progress saved ({len(results)} total)")
        
        # Small delay to avoid rate limits
        time.sleep(0.5)
    
    # Final save
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    
    # Summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    for level in [1, 2, 3, 4]:
        level_results = [r for r in results if r["level"] == level]
        correct = sum(1 for r in level_results if r["success"])
        total = len(level_results)
        pct = correct/total*100 if total > 0 else 0
        print(f"Level {level}: {correct}/{total} ({pct:.1f}%)")
    
    total_correct = sum(1 for r in results if r["success"])
    total = len(results)
    print(f"\nTOTAL: {total_correct}/{total} ({total_correct/total*100:.1f}%)")
    print(f"Total tokens: {total_tokens:,}")
    
    # Cost estimate
    cost = (total_tokens / 1_000_000) * 0.75  # avg of in/out pricing
    print(f"Estimated cost: ${cost:.3f}")

if __name__ == "__main__":
    main()
