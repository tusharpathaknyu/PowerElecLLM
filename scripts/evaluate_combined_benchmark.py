#!/usr/bin/env python3
"""
Evaluate Combined Benchmark (400 train / 100 test)

Handles both:
- Expert-verified (conceptual): MCQ answer extraction + reasoning quality
- Synthetic (SPICE): Circuit simulation with waveform analysis

Usage:
    python scripts/evaluate_combined_benchmark.py --model gpt-4o --split test
    python scripts/evaluate_combined_benchmark.py --model gpt-4o --split test --num 20
"""

import argparse
import json
import os
import sys
import re
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

BENCHMARK_DIR = PROJECT_ROOT / "benchmarks" / "combined_benchmark"
RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "combined_benchmark_results"

# Try to import SPICE evaluator
try:
    from spice_evaluator import PowerConverterEvaluator
    SPICE_AVAILABLE = True
except ImportError:
    SPICE_AVAILABLE = False
    print("⚠️  SPICE evaluator not available")


@dataclass
class EvalResult:
    """Evaluation result for a single problem"""
    problem_id: str
    level: int
    benchmark_type: str  # expert_verified or synthetic
    evaluation_method: str  # conceptual or spice
    
    # Common fields
    success: bool
    score: float
    latency_ms: float
    
    # Conceptual evaluation
    answer_extracted: Optional[str] = None
    reasoning_quality: str = "unknown"
    
    # SPICE evaluation
    target_vout: Optional[float] = None
    predicted_vout: Optional[float] = None
    simulated_vout: Optional[float] = None
    vout_error_pct: Optional[float] = None
    ripple_pct: Optional[float] = None
    efficiency_pct: Optional[float] = None
    spice_success: bool = False
    
    error_msg: Optional[str] = None


class CombinedEvaluator:
    """Evaluator for combined benchmark"""
    
    def __init__(self, model: str):
        self.model = model
        self._setup_llm_client()
        
        if SPICE_AVAILABLE:
            self.spice_evaluator = PowerConverterEvaluator()
        else:
            self.spice_evaluator = None
        
        self.system_prompt_conceptual = """You are an expert in power electronics.
For multiple choice questions, analyze the options and state your answer clearly as "The answer is (X)".
For calculation problems, show your work step by step and provide the numerical answer with units.
Be precise and use standard power electronics terminology."""

        self.system_prompt_design = """You are an expert power electronics engineer.
When given a converter design problem:
1. Identify the topology (buck, boost, buck-boost, etc.)
2. Calculate the duty cycle using the correct equation
3. Calculate inductor and capacitor values for the given ripple requirements
4. Verify your answer

Always provide numerical values for:
- Output voltage (Vout) in volts
- Duty cycle (D) as a decimal (e.g., 0.5 not 50%)
- Inductor value (L) with units (e.g., 100µH)
- Capacitor value (C) with units (e.g., 47µF)

Format your final answer clearly with these values."""

    def _setup_llm_client(self):
        """Setup LLM client based on model name"""
        from openai import OpenAI
        
        if "grok" in self.model.lower():
            self.client = OpenAI(
                api_key=os.environ.get("XAI_API_KEY"),
                base_url="https://api.x.ai/v1"
            )
        elif "llama" in self.model.lower() or "mixtral" in self.model.lower():
            self.client = OpenAI(
                api_key=os.environ.get("GROQ_API_KEY"),
                base_url="https://api.groq.com/openai/v1"
            )
        elif "gemini" in self.model.lower():
            import google.generativeai as genai
            genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
            self.gemini_model = genai.GenerativeModel(self.model)
            self.client = None
        else:
            self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    def call_llm(self, prompt: str, system_prompt: str) -> Tuple[str, float]:
        """Call LLM and return response with latency"""
        start = time.time()
        
        try:
            if hasattr(self, 'gemini_model') and self.client is None:
                response = self.gemini_model.generate_content(prompt)
                text = response.text
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                    max_tokens=2000
                )
                text = response.choices[0].message.content
            
            latency = (time.time() - start) * 1000
            return text, latency
        except Exception as e:
            latency = (time.time() - start) * 1000
            return f"ERROR: {str(e)}", latency
    
    def evaluate_conceptual(self, problem: Dict) -> EvalResult:
        """Evaluate conceptual/MCQ problem"""
        problem_text = problem.get("problem_text", problem.get("prompt", ""))
        
        response, latency = self.call_llm(problem_text, self.system_prompt_conceptual)
        
        # Extract answer
        answer = None
        mcq_match = re.search(r"answer\s*(?:is|:)?\s*\(?([A-D])\)?", response, re.IGNORECASE)
        if mcq_match:
            answer = mcq_match.group(1).upper()
        
        # Check reasoning quality
        reasoning_indicators = ["because", "therefore", "since", "thus", "formula", "equation"]
        reasoning_count = sum(1 for ind in reasoning_indicators if ind in response.lower())
        
        if reasoning_count >= 3:
            reasoning_quality = "good"
        elif reasoning_count >= 1:
            reasoning_quality = "partial"
        else:
            reasoning_quality = "poor"
        
        # Score based on response completeness
        if answer and reasoning_quality == "good":
            score = 100
            success = True
        elif answer:
            score = 70
            success = True
        elif len(response) > 200:
            score = 40
            success = False
        else:
            score = 0
            success = False
        
        return EvalResult(
            problem_id=problem.get("id", "unknown"),
            level=problem.get("level", 0),
            benchmark_type="expert_verified",
            evaluation_method="conceptual",
            success=success,
            score=score,
            latency_ms=latency,
            answer_extracted=answer,
            reasoning_quality=reasoning_quality
        )
    
    def evaluate_spice(self, problem: Dict) -> EvalResult:
        """Evaluate design problem with SPICE simulation"""
        prompt = problem.get("prompt", "")
        specs = problem.get("specs", {})
        
        response, latency = self.call_llm(prompt, self.system_prompt_design)
        
        # Parse LLM response for component values
        parsed = self._parse_design_response(response)
        
        # Get target values
        target_vout = specs.get("vout")
        predicted_vout = parsed.get("vout")
        
        # Simple evaluation if SPICE not available
        if not self.spice_evaluator or not SPICE_AVAILABLE:
            if predicted_vout and target_vout:
                error_pct = abs(predicted_vout - target_vout) / target_vout * 100
                success = error_pct < 10
                score = max(0, 100 - error_pct * 5)
            else:
                error_pct = 100
                success = False
                score = 0
            
            return EvalResult(
                problem_id=problem.get("id", "unknown"),
                level=problem.get("level", 0),
                benchmark_type="synthetic",
                evaluation_method="spice",
                success=success,
                score=score,
                latency_ms=latency,
                target_vout=target_vout,
                predicted_vout=predicted_vout,
                vout_error_pct=error_pct,
                spice_success=False,
                error_msg="SPICE not available - using simple Vout comparison"
            )
        
        # Full SPICE evaluation
        try:
            topology = parsed.get("topology") or specs.get("topology", "buck")
            
            components = {
                "D": parsed.get("duty", self._calc_duty(topology, specs)),
                "L": parsed.get("L", 100e-6),
                "C": parsed.get("C", 100e-6),
                "R_load": specs.get("vout", 12)**2 / specs.get("power", 100)
            }
            
            spice_results, eval_score = self.spice_evaluator.evaluate(
                topology, components, specs, problem.get("level", 1)
            )
            
            return EvalResult(
                problem_id=problem.get("id", "unknown"),
                level=problem.get("level", 0),
                benchmark_type="synthetic",
                evaluation_method="spice",
                success=eval_score.total_score >= 60,
                score=eval_score.total_score,
                latency_ms=latency,
                target_vout=target_vout,
                predicted_vout=predicted_vout,
                simulated_vout=spice_results.vout_dc,
                vout_error_pct=eval_score.details.get("vout_error_pct", 100),
                ripple_pct=spice_results.vout_ripple_pct * 100,
                efficiency_pct=spice_results.efficiency * 100,
                spice_success=spice_results.simulation_success
            )
        except Exception as e:
            return EvalResult(
                problem_id=problem.get("id", "unknown"),
                level=problem.get("level", 0),
                benchmark_type="synthetic",
                evaluation_method="spice",
                success=False,
                score=0,
                latency_ms=latency,
                target_vout=target_vout,
                predicted_vout=predicted_vout,
                spice_success=False,
                error_msg=str(e)
            )
    
    def _parse_design_response(self, response: str) -> Dict:
        """Parse LLM response for design values"""
        result = {"vout": None, "duty": None, "L": None, "C": None, "topology": None}
        
        # Parse Vout
        for pattern in [r"[Vv]out\s*[=:≈]\s*([\d.]+)\s*V", r"output.*?([\d.]+)\s*V"]:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                result["vout"] = float(match.group(1))
                break
        
        # Parse duty cycle
        for pattern in [r"[Dd]uty.*?[=:]\s*([\d.]+)%", r"[Dd]\s*[=:]\s*([\d.]+)"]:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                val = float(match.group(1))
                result["duty"] = val / 100 if val > 1 else val
                break
        
        # Parse inductor
        mult = {"H": 1, "mH": 1e-3, "µH": 1e-6, "uH": 1e-6}
        for pattern in [r"L\s*[=:]\s*([\d.]+)\s*(µH|uH|mH|H)"]:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                result["L"] = float(match.group(1)) * mult.get(match.group(2), 1e-6)
                break
        
        # Parse capacitor
        cmult = {"F": 1, "mF": 1e-3, "µF": 1e-6, "uF": 1e-6, "nF": 1e-9}
        for pattern in [r"C\s*[=:]\s*([\d.]+)\s*(µF|uF|mF|F|nF)"]:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                result["C"] = float(match.group(1)) * cmult.get(match.group(2), 1e-6)
                break
        
        # Detect topology
        resp_lower = response.lower()
        if "buck-boost" in resp_lower:
            result["topology"] = "buck_boost"
        elif "boost" in resp_lower:
            result["topology"] = "boost"
        elif "buck" in resp_lower:
            result["topology"] = "buck"
        
        return result
    
    def _calc_duty(self, topology: str, specs: Dict) -> float:
        """Calculate expected duty cycle"""
        vin = specs.get("vin", 24)
        vout = specs.get("vout", 12)
        if topology == "buck":
            return vout / vin
        elif topology == "boost":
            return 1 - vin / vout
        else:
            return vout / (vin + vout)
    
    def evaluate_problem(self, problem: Dict) -> EvalResult:
        """Evaluate a single problem based on its type"""
        eval_method = problem.get("evaluation_method", "conceptual")
        
        if eval_method == "spice":
            return self.evaluate_spice(problem)
        else:
            return self.evaluate_conceptual(problem)
    
    def run_evaluation(self, problems: List[Dict], progress: bool = True) -> List[EvalResult]:
        """Run evaluation on all problems"""
        results = []
        total = len(problems)
        
        for i, problem in enumerate(problems, 1):
            if progress:
                pid = problem.get("id", "?")[:20]
                ptype = problem.get("evaluation_method", "?")[:4]
                print(f"[{i}/{total}] {pid} ({ptype})... ", end="", flush=True)
            
            result = self.evaluate_problem(problem)
            results.append(result)
            
            if progress:
                status = "✅" if result.success else "❌"
                print(f"{status} (score: {result.score:.0f}, {result.latency_ms:.0f}ms)")
            
            # Rate limiting
            time.sleep(0.3)
        
        return results


def load_benchmark(split: str = "test") -> List[Dict]:
    """Load benchmark problems"""
    filename = "test_100.json" if split == "test" else "train_400.json"
    filepath = BENCHMARK_DIR / filename
    
    with open(filepath) as f:
        data = json.load(f)
    
    return data.get("problems", [])


def save_results(results: List[EvalResult], model: str, split: str):
    """Save evaluation results"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_safe = model.replace(":", "_").replace("/", "_")
    filename = f"eval_{model_safe}_{split}_{timestamp}.json"
    
    # Calculate summary
    total = len(results)
    passed = sum(1 for r in results if r.success)
    avg_score = sum(r.score for r in results) / total if total > 0 else 0
    avg_latency = sum(r.latency_ms for r in results) / total if total > 0 else 0
    
    # By type
    conceptual = [r for r in results if r.evaluation_method == "conceptual"]
    spice = [r for r in results if r.evaluation_method == "spice"]
    
    conceptual_pass = sum(1 for r in conceptual if r.success)
    spice_pass = sum(1 for r in spice if r.success)
    
    output = {
        "metadata": {
            "model": model,
            "split": split,
            "timestamp": timestamp,
            "total_problems": total
        },
        "summary": {
            "total_passed": passed,
            "pass_rate": passed / total * 100 if total > 0 else 0,
            "avg_score": avg_score,
            "avg_latency_ms": avg_latency,
            "conceptual": {
                "total": len(conceptual),
                "passed": conceptual_pass,
                "pass_rate": conceptual_pass / len(conceptual) * 100 if conceptual else 0
            },
            "spice": {
                "total": len(spice),
                "passed": spice_pass,
                "pass_rate": spice_pass / len(spice) * 100 if spice else 0
            }
        },
        "results": [asdict(r) for r in results]
    }
    
    filepath = RESULTS_DIR / filename
    with open(filepath, "w") as f:
        json.dump(output, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Model: {model}")
    print(f"Split: {split}")
    print(f"Total: {total} problems")
    print(f"\nOverall: {passed}/{total} passed ({passed/total*100:.1f}%)")
    print(f"Average Score: {avg_score:.1f}")
    print(f"Average Latency: {avg_latency:.0f}ms")
    print(f"\nConceptual (Expert): {conceptual_pass}/{len(conceptual)} ({conceptual_pass/len(conceptual)*100:.1f}%)" if conceptual else "")
    print(f"SPICE (Design): {spice_pass}/{len(spice)} ({spice_pass/len(spice)*100:.1f}%)" if spice else "")
    print(f"\nResults saved: {filepath}")
    
    return filepath


def main():
    parser = argparse.ArgumentParser(description="Evaluate Combined Benchmark")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model to evaluate")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--num", type=int, help="Number of problems (default: all)")
    
    args = parser.parse_args()
    
    # Load problems
    problems = load_benchmark(args.split)
    if args.num:
        problems = problems[:args.num]
    
    print(f"\n{'='*60}")
    print(f"Combined Benchmark Evaluation")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Split: {args.split}")
    print(f"Problems: {len(problems)}")
    print(f"{'='*60}\n")
    
    # Run evaluation
    evaluator = CombinedEvaluator(args.model)
    results = evaluator.run_evaluation(problems)
    
    # Save results
    save_results(results, args.model, args.split)


if __name__ == "__main__":
    main()
