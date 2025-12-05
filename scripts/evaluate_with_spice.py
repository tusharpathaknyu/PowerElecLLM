#!/usr/bin/env python3
"""
Enhanced LLM Evaluation with SPICE-Based Multi-Criteria Scoring

For L1/L2: Uses simple Vout accuracy check (±5%)
For L3/L4: Uses full SPICE simulation with multi-criteria scoring:
  - Vout accuracy
  - Output ripple
  - Efficiency estimate
  - Current ripple
  - Component stress

Cost estimate per model (150 problems):
  - GPT-4o: ~$0.50
  - GPT-4o-mini: ~$0.05
  - Grok: ~$0.30
  - LLaMA (Groq): FREE
"""

import os
import sys
import json
import re
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from openai import OpenAI
from spice_evaluator import PowerConverterEvaluator, SpiceResults, EvaluationScore


class EnhancedLLMEvaluator:
    """
    LLM evaluator with SPICE-based verification for L3/L4
    """
    
    def __init__(self, model: str, use_spice_for_all: bool = False):
        self.model = model
        self.use_spice_for_all = use_spice_for_all
        self.spice_evaluator = PowerConverterEvaluator()
        
        # Detect which API to use
        if "grok" in model.lower():
            self.api_type = "xai"
            self.client = OpenAI(
                api_key=os.environ.get("XAI_API_KEY"),
                base_url="https://api.x.ai/v1"
            )
        elif "llama" in model.lower() or "mixtral" in model.lower():
            self.api_type = "groq"
            self.client = OpenAI(
                api_key=os.environ.get("GROQ_API_KEY"),
                base_url="https://api.groq.com/openai/v1"
            )
        else:
            self.api_type = "openai"
            self.client = OpenAI()
        
        # System prompt for power electronics
        self.system_prompt = """You are an expert power electronics engineer. 
When given a converter design problem:
1. Identify the topology (buck, boost, buck-boost, etc.)
2. Calculate the duty cycle using the correct equation
3. Calculate inductor and capacitor values
4. Verify your answer

Always provide numerical values for:
- Output voltage (Vout)
- Duty cycle (D) as a decimal (e.g., 0.5 not 50%)
- Inductor value (L) with units
- Capacitor value (C) with units

Format your final answer clearly with these values."""

    def call_llm(self, prompt: str) -> str:
        """Call the LLM and get response"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"API Error: {e}")
            return ""
    
    def parse_llm_response(self, response: str) -> Dict:
        """Extract component values from LLM response"""
        result = {
            "vout": None,
            "duty": None,
            "L": None,
            "C": None,
            "topology": None
        }
        
        # Parse Vout
        vout_patterns = [
            r"[Vv]out\s*[=:≈]\s*([\d.]+)\s*V",
            r"output voltage[=:\s]*([\d.]+)\s*V",
            r"([\d.]+)\s*V\s*(?:output|DC)",
        ]
        for pattern in vout_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                result["vout"] = float(match.group(1))
                break
        
        # Parse duty cycle
        duty_patterns = [
            r"[Dd]uty\s*(?:cycle)?\s*[=:≈]\s*([\d.]+)%",
            r"[Dd]\s*[=:]\s*([\d.]+)",
            r"duty\s*[=:]\s*([\d.]+)",
        ]
        for pattern in duty_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                val = float(match.group(1))
                result["duty"] = val / 100 if val > 1 else val
                break
        
        # Parse inductor
        L_patterns = [
            r"L\s*[=:]\s*([\d.]+)\s*(µH|uH|mH|H)",
            r"[Ii]nductor\s*[=:]\s*([\d.]+)\s*(µH|uH|mH|H)",
        ]
        multipliers = {"H": 1, "mH": 1e-3, "µH": 1e-6, "uH": 1e-6}
        for pattern in L_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                val = float(match.group(1))
                unit = match.group(2)
                result["L"] = val * multipliers.get(unit, 1e-6)
                break
        
        # Parse capacitor
        C_patterns = [
            r"C\s*[=:]\s*([\d.]+)\s*(µF|uF|mF|F|nF|pF)",
            r"[Cc]apacitor\s*[=:]\s*([\d.]+)\s*(µF|uF|mF|F|nF|pF)",
        ]
        c_mult = {"F": 1, "mF": 1e-3, "µF": 1e-6, "uF": 1e-6, "nF": 1e-9, "pF": 1e-12}
        for pattern in C_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                val = float(match.group(1))
                unit = match.group(2)
                result["C"] = val * c_mult.get(unit, 1e-6)
                break
        
        # Detect topology
        response_lower = response.lower()
        if "buck-boost" in response_lower or "buck boost" in response_lower:
            result["topology"] = "buck_boost"
        elif "boost" in response_lower:
            result["topology"] = "boost"
        elif "buck" in response_lower:
            result["topology"] = "buck"
        
        return result
    
    def evaluate_simple(self, problem: Dict, llm_response: str, parsed: Dict) -> Dict:
        """Simple evaluation - just check Vout accuracy (for L1/L2)"""
        target_vout = problem.get("specs", {}).get("vout")
        predicted_vout = parsed.get("vout")
        
        if target_vout is None or predicted_vout is None:
            return {
                "success": False,
                "error": "Could not parse Vout",
                "predicted_vout": predicted_vout,
                "target_vout": target_vout
            }
        
        error_pct = abs(predicted_vout - target_vout) / target_vout * 100
        success = error_pct < 5  # 5% tolerance
        
        return {
            "success": success,
            "predicted_vout": predicted_vout,
            "target_vout": target_vout,
            "error_pct": error_pct,
            "score": 100 if success else max(0, 100 - error_pct * 5)
        }
    
    def evaluate_with_spice(self, problem: Dict, llm_response: str, parsed: Dict) -> Dict:
        """
        Full SPICE evaluation with multi-criteria scoring (for L3/L4)
        """
        specs = problem.get("specs", {})
        topology = parsed.get("topology") or problem.get("topology", "buck")
        
        # Build component dict
        components = {}
        if parsed.get("duty") is not None:
            components["D"] = parsed["duty"]
        else:
            # Calculate expected duty from topology
            vin = specs.get("vin", 24)
            vout = specs.get("vout", 12)
            if topology == "buck":
                components["D"] = vout / vin
            elif topology == "boost":
                components["D"] = 1 - vin / vout
            else:
                components["D"] = vout / (vin + vout)
        
        if parsed.get("L"):
            components["L"] = parsed["L"]
        else:
            components["L"] = 100e-6  # Default
        
        if parsed.get("C"):
            components["C"] = parsed["C"]
        else:
            components["C"] = 100e-6  # Default
        
        # Calculate load resistance from specs
        power = specs.get("power", 100)
        vout = specs.get("vout", 12)
        components["R_load"] = vout**2 / power if power > 0 else 10
        
        # Run SPICE simulation
        level = problem.get("level", 3)
        try:
            spice_results, score = self.spice_evaluator.evaluate(
                topology, components, specs, level
            )
            
            return {
                "success": score.total_score >= 60,  # Pass threshold
                "spice_success": spice_results.simulation_success,
                "total_score": score.total_score,
                "vout_score": score.vout_score,
                "ripple_score": score.ripple_score,
                "efficiency_score": score.efficiency_score,
                "current_score": score.current_score,
                "stress_score": score.stress_score,
                "simulated_vout": spice_results.vout_dc,
                "target_vout": specs.get("vout"),
                "vout_error_pct": score.details.get("vout_error_pct", 100),
                "ripple_pct": spice_results.vout_ripple_pct * 100,
                "efficiency_pct": spice_results.efficiency * 100,
                "ccm_mode": score.operation_mode_correct,
                "components_used": components,
                "error": spice_results.error_message if not spice_results.simulation_success else None
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "total_score": 0
            }
    
    def evaluate_problem(self, problem: Dict) -> Dict:
        """Evaluate a single problem"""
        prompt = problem.get("prompt", "")
        level = problem.get("level", 1)
        
        # Call LLM
        start_time = time.time()
        response = self.call_llm(prompt)
        latency = time.time() - start_time
        
        if not response:
            return {
                "problem_id": problem.get("id"),
                "level": level,
                "success": False,
                "error": "No LLM response",
                "latency": latency
            }
        
        # Parse response
        parsed = self.parse_llm_response(response)
        
        # Choose evaluation method based on level
        if level >= 3 or self.use_spice_for_all:
            eval_result = self.evaluate_with_spice(problem, response, parsed)
        else:
            eval_result = self.evaluate_simple(problem, response, parsed)
        
        eval_result.update({
            "problem_id": problem.get("id"),
            "level": level,
            "latency": latency,
            "parsed_values": parsed,
            "llm_response_length": len(response)
        })
        
        return eval_result
    
    def run_evaluation(
        self,
        problems: List[Dict],
        output_file: str = None,
        progress_callback=None
    ) -> Dict:
        """Run evaluation on all problems"""
        
        results = []
        total = len(problems)
        
        for i, problem in enumerate(problems):
            result = self.evaluate_problem(problem)
            results.append(result)
            
            if progress_callback:
                progress_callback(i + 1, total, result)
            
            # Rate limiting
            if self.api_type == "groq":
                time.sleep(1)  # Groq rate limit
            else:
                time.sleep(0.2)
        
        # Calculate summary statistics
        summary = self._calculate_summary(results)
        
        output = {
            "model": self.model,
            "timestamp": datetime.now().isoformat(),
            "total_problems": total,
            "summary": summary,
            "results": results
        }
        
        if output_file:
            with open(output_file, "w") as f:
                json.dump(output, f, indent=2)
        
        return output
    
    def _calculate_summary(self, results: List[Dict]) -> Dict:
        """Calculate summary statistics"""
        
        by_level = {}
        for r in results:
            lvl = r.get("level", 1)
            if lvl not in by_level:
                by_level[lvl] = {"total": 0, "passed": 0, "scores": []}
            by_level[lvl]["total"] += 1
            if r.get("success"):
                by_level[lvl]["passed"] += 1
            if "total_score" in r:
                by_level[lvl]["scores"].append(r["total_score"])
            elif "score" in r:
                by_level[lvl]["scores"].append(r["score"])
        
        summary = {
            "total": len(results),
            "passed": sum(1 for r in results if r.get("success")),
            "by_level": {}
        }
        
        for lvl, data in by_level.items():
            summary["by_level"][f"L{lvl}"] = {
                "total": data["total"],
                "passed": data["passed"],
                "accuracy": data["passed"] / data["total"] * 100 if data["total"] > 0 else 0,
                "avg_score": sum(data["scores"]) / len(data["scores"]) if data["scores"] else 0
            }
        
        summary["overall_accuracy"] = summary["passed"] / summary["total"] * 100 if summary["total"] > 0 else 0
        
        return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM with SPICE verification")
    parser.add_argument("--model", required=True, help="Model to evaluate")
    parser.add_argument("--test-file", default="benchmarks/test_set/test_problems.json",
                       help="Test problems JSON file")
    parser.add_argument("--output", help="Output results file")
    parser.add_argument("--level", type=int, help="Only evaluate specific level")
    parser.add_argument("--num", type=int, help="Number of problems to evaluate")
    parser.add_argument("--spice-all", action="store_true", 
                       help="Use SPICE for all levels (not just L3/L4)")
    args = parser.parse_args()
    
    # Load problems
    test_file = Path(args.test_file)
    if not test_file.exists():
        print(f"Error: Test file not found: {test_file}")
        sys.exit(1)
    
    with open(test_file) as f:
        data = json.load(f)
    
    problems = data.get("problems", [])
    
    # Filter by level if specified
    if args.level:
        problems = [p for p in problems if p.get("level") == args.level]
    
    # Limit number if specified
    if args.num:
        problems = problems[:args.num]
    
    print(f"=== Evaluating {args.model} ===")
    print(f"Problems: {len(problems)}")
    print(f"SPICE verification: L3/L4" + (" + L1/L2" if args.spice_all else ""))
    print()
    
    # Create evaluator
    evaluator = EnhancedLLMEvaluator(args.model, use_spice_for_all=args.spice_all)
    
    # Progress callback
    def progress(current, total, result):
        status = "✓" if result.get("success") else "✗"
        score = result.get("total_score", result.get("score", "N/A"))
        if isinstance(score, float):
            score = f"{score:.1f}"
        print(f"[{current}/{total}] {result.get('problem_id', 'N/A')} {status} Score: {score}")
    
    # Run evaluation
    output_file = args.output or f"benchmarks/results/spice_eval_{args.model.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    results = evaluator.run_evaluation(problems, output_file, progress_callback=progress)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    summary = results["summary"]
    print(f"Total: {summary['passed']}/{summary['total']} ({summary['overall_accuracy']:.1f}%)")
    print()
    for level, data in sorted(summary["by_level"].items()):
        print(f"  {level}: {data['passed']}/{data['total']} ({data['accuracy']:.1f}%) - Avg Score: {data['avg_score']:.1f}")
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
