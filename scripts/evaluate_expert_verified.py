#!/usr/bin/env python3
"""
Evaluate LLM on Expert-Verified Test Set

This script evaluates models on the 20% held-out test set from expert-verified
problems (GATE + MIT 6.334). Results are saved separately from existing benchmarks.

Usage:
    python scripts/evaluate_expert_verified.py --level 1 --model gpt-4o
    python scripts/evaluate_expert_verified.py --all --model gpt-4o
"""

import argparse
import json
import os
import sys
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Expert verified paths
EXPERT_DIR = PROJECT_ROOT / "benchmarks" / "expert_verified"
TRAIN_TEST_DIR = EXPERT_DIR / "train_test_split"
RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "expert_verified_results"


@dataclass
class ExpertEvalResult:
    """Result of evaluating a single expert-verified problem"""
    problem_id: str
    source: str  # GATE or MIT
    year: Optional[int]
    level: int
    difficulty: str
    problem_type: str
    problem_text: str
    
    # LLM Response
    llm_response: str
    
    # Evaluation
    response_quality: str  # complete, partial, incorrect, no_response
    answer_extracted: Optional[str] = None
    reasoning_quality: str = "unknown"  # good, partial, poor
    
    # Metadata
    model: str = ""
    latency_ms: float = 0
    error_msg: Optional[str] = None


class ExpertTestLoader:
    """Load expert-verified test problems"""
    
    def __init__(self):
        self.train_test_dir = TRAIN_TEST_DIR
        
    def load_test_level(self, level: int) -> List[Dict]:
        """Load test problems for a specific level"""
        filepath = self.train_test_dir / f"level_{level}_test.json"
        if not filepath.exists():
            print(f"❌ Test file not found: {filepath}")
            return []
        
        with open(filepath) as f:
            data = json.load(f)
        return data.get("problems", [])
    
    def load_all_test(self) -> List[Dict]:
        """Load all test problems"""
        filepath = self.train_test_dir / "combined_test.json"
        if not filepath.exists():
            print(f"❌ Combined test file not found: {filepath}")
            return []
        
        with open(filepath) as f:
            data = json.load(f)
        return data.get("problems", [])
    
    def load_train_level(self, level: int) -> List[Dict]:
        """Load train problems for a specific level (for few-shot examples)"""
        filepath = self.train_test_dir / f"level_{level}_train.json"
        if not filepath.exists():
            return []
        
        with open(filepath) as f:
            data = json.load(f)
        return data.get("problems", [])


class LLMClient:
    """Unified LLM client supporting multiple providers"""
    
    def __init__(self, model: str, api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._client = None
        self._setup_client()
    
    def _setup_client(self):
        if "gpt" in self.model or self.model.startswith("ft:"):
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
            self.provider = "openai"
        elif "gemini" in self.model:
            import google.generativeai as genai
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY") or self.api_key)
            self._client = genai.GenerativeModel(self.model)
            self.provider = "google"
        elif "grok" in self.model:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=os.getenv("XAI_API_KEY") or self.api_key,
                base_url="https://api.x.ai/v1"
            )
            self.provider = "xai"
        elif "llama" in self.model or "mixtral" in self.model:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=os.getenv("GROQ_API_KEY") or self.api_key,
                base_url="https://api.groq.com/openai/v1"
            )
            self.provider = "groq"
        else:
            raise ValueError(f"Unknown model: {self.model}")
    
    def query(self, prompt: str, system_prompt: str = None) -> tuple:
        """Query the LLM and return (response, latency_ms)"""
        import time
        start = time.time()
        
        try:
            if self.provider == "google":
                response = self._client.generate_content(prompt)
                text = response.text
            else:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})
                
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=2000
                )
                text = response.choices[0].message.content
            
            latency = (time.time() - start) * 1000
            return text, latency
            
        except Exception as e:
            latency = (time.time() - start) * 1000
            return f"ERROR: {str(e)}", latency


def create_prompt_for_problem(problem: Dict, few_shot_examples: List[Dict] = None) -> str:
    """Create a prompt for evaluating a power electronics problem"""
    
    # System context
    prompt = """You are an expert in power electronics. Solve the following problem step by step.

For multiple choice questions, state your answer clearly (e.g., "The answer is (A)").
For numerical problems, show your calculations and state the final answer with units.

"""
    
    # Add few-shot examples if provided
    if few_shot_examples:
        prompt += "Here are some example problems and solutions:\n\n"
        for i, ex in enumerate(few_shot_examples[:2], 1):
            prompt += f"Example {i}:\n{ex.get('problem_text', '')}\n\n"
    
    # The actual problem
    prompt += "Now solve this problem:\n\n"
    prompt += problem.get("problem_text", "No problem text available")
    
    return prompt


def evaluate_response(problem: Dict, response: str) -> Dict:
    """Evaluate the quality of an LLM response"""
    
    result = {
        "response_quality": "no_response",
        "answer_extracted": None,
        "reasoning_quality": "unknown"
    }
    
    if not response or "ERROR" in response:
        return result
    
    response_lower = response.lower()
    
    # Check for MCQ answer
    mcq_match = re.search(r"answer\s*(?:is|:)?\s*\(?([A-D])\)?", response, re.IGNORECASE)
    if mcq_match:
        result["answer_extracted"] = mcq_match.group(1).upper()
        result["response_quality"] = "complete"
    
    # Check for numerical answer
    num_match = re.search(r"(?:answer|result|output|=)\s*[:\s]*(-?[\d.]+)\s*([VAWHz%°Ω])?", response, re.IGNORECASE)
    if num_match:
        result["answer_extracted"] = num_match.group(0)
        result["response_quality"] = "complete"
    
    # Reasoning quality
    reasoning_indicators = [
        "because", "therefore", "since", "thus",
        "we can", "this means", "applying", "using",
        "formula", "equation", "calculate"
    ]
    
    reasoning_count = sum(1 for ind in reasoning_indicators if ind in response_lower)
    if reasoning_count >= 3:
        result["reasoning_quality"] = "good"
    elif reasoning_count >= 1:
        result["reasoning_quality"] = "partial"
    else:
        result["reasoning_quality"] = "poor"
    
    # If response is long but no clear answer
    if result["response_quality"] == "no_response" and len(response) > 100:
        result["response_quality"] = "partial"
    
    return result


def run_evaluation(
    model: str,
    level: Optional[int] = None,
    num_problems: Optional[int] = None,
    use_few_shot: bool = True
) -> List[ExpertEvalResult]:
    """Run evaluation on expert-verified test set"""
    
    loader = ExpertTestLoader()
    client = LLMClient(model)
    
    # Load test problems
    if level:
        problems = loader.load_test_level(level)
        few_shot_examples = loader.load_train_level(level) if use_few_shot else []
    else:
        problems = loader.load_all_test()
        few_shot_examples = []  # Mixed levels, skip few-shot
    
    if num_problems:
        problems = problems[:num_problems]
    
    print(f"\n{'='*70}")
    print(f"Expert-Verified Benchmark Evaluation")
    print(f"Model: {model}")
    print(f"Level: {level or 'All'}")
    print(f"Problems: {len(problems)}")
    print(f"Few-shot: {use_few_shot}")
    print(f"{'='*70}\n")
    
    results = []
    
    for i, problem in enumerate(problems, 1):
        problem_id = problem.get("id", f"unknown_{i}")
        print(f"[{i}/{len(problems)}] {problem_id}... ", end="", flush=True)
        
        # Create prompt
        prompt = create_prompt_for_problem(problem, few_shot_examples if use_few_shot else None)
        
        # Query LLM
        response, latency = client.query(prompt)
        
        # Evaluate response
        eval_result = evaluate_response(problem, response)
        
        # Create result object
        result = ExpertEvalResult(
            problem_id=problem_id,
            source=problem.get("source", "unknown"),
            year=problem.get("year"),
            level=problem.get("level", 0),
            difficulty=problem.get("difficulty", "unknown"),
            problem_type=problem.get("problem_type", "general"),
            problem_text=problem.get("problem_text", "")[:500],
            llm_response=response[:2000],
            response_quality=eval_result["response_quality"],
            answer_extracted=eval_result["answer_extracted"],
            reasoning_quality=eval_result["reasoning_quality"],
            model=model,
            latency_ms=latency
        )
        results.append(result)
        
        # Print status
        status = "✅" if eval_result["response_quality"] == "complete" else "⚠️" if eval_result["response_quality"] == "partial" else "❌"
        print(f"{status} ({latency:.0f}ms)")
    
    return results


def save_results(results: List[ExpertEvalResult], model: str, level: Optional[int]):
    """Save evaluation results"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    level_str = f"L{level}" if level else "all"
    model_safe = model.replace(":", "_").replace("/", "_")
    
    filename = f"expert_eval_{model_safe}_{level_str}_{timestamp}.json"
    filepath = RESULTS_DIR / filename
    
    # Calculate summary stats
    total = len(results)
    complete = sum(1 for r in results if r.response_quality == "complete")
    partial = sum(1 for r in results if r.response_quality == "partial")
    good_reasoning = sum(1 for r in results if r.reasoning_quality == "good")
    avg_latency = sum(r.latency_ms for r in results) / total if total > 0 else 0
    
    output = {
        "metadata": {
            "model": model,
            "level": level,
            "timestamp": timestamp,
            "total_problems": total,
            "benchmark_type": "expert_verified_test_set"
        },
        "summary": {
            "complete_answers": complete,
            "partial_answers": partial,
            "completion_rate": complete / total * 100 if total > 0 else 0,
            "good_reasoning_rate": good_reasoning / total * 100 if total > 0 else 0,
            "avg_latency_ms": avg_latency
        },
        "results": [asdict(r) for r in results]
    }
    
    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n{'='*70}")
    print("EVALUATION SUMMARY")
    print(f"{'='*70}")
    print(f"Total Problems: {total}")
    print(f"Complete Answers: {complete} ({complete/total*100:.1f}%)" if total > 0 else "Complete Answers: 0")
    print(f"Partial Answers: {partial} ({partial/total*100:.1f}%)" if total > 0 else "Partial Answers: 0")
    print(f"Good Reasoning: {good_reasoning} ({good_reasoning/total*100:.1f}%)" if total > 0 else "Good Reasoning: 0")
    print(f"Avg Latency: {avg_latency:.0f}ms")
    print(f"\nResults saved to: {filepath}")
    
    return filepath


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM on Expert-Verified Test Set")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model to evaluate")
    parser.add_argument("--level", type=int, choices=[1, 2, 3, 4], help="Difficulty level (1-4)")
    parser.add_argument("--all", action="store_true", help="Evaluate all levels combined")
    parser.add_argument("--num", type=int, help="Number of problems to evaluate")
    parser.add_argument("--no-few-shot", action="store_true", help="Disable few-shot examples")
    
    args = parser.parse_args()
    
    # Validate args
    if not args.level and not args.all:
        print("Please specify --level (1-4) or --all")
        sys.exit(1)
    
    level = None if args.all else args.level
    use_few_shot = not args.no_few_shot
    
    # Run evaluation
    results = run_evaluation(
        model=args.model,
        level=level,
        num_problems=args.num,
        use_few_shot=use_few_shot
    )
    
    # Save results
    save_results(results, args.model, level)


if __name__ == "__main__":
    main()
