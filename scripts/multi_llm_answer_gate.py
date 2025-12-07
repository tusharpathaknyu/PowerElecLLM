#!/usr/bin/env python3
"""
Multi-LLM Answer Generator for GATE Power Electronics Questions
Uses multiple LLM models to answer questions and finds consensus.
"""

import json
import os
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from openai import OpenAI

# Try to import optional providers
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Initialize clients
openai_client = OpenAI()

# Initialize Gemini if available
if GEMINI_AVAILABLE and os.getenv("GOOGLE_API_KEY"):
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Models to use for consensus
MODELS = [
    {"provider": "openai", "model": "gpt-4o", "name": "GPT-4o"},
    {"provider": "openai", "model": "gpt-4o-mini", "name": "GPT-4o-mini"},
    {"provider": "openai", "model": "gpt-4-turbo", "name": "GPT-4-Turbo"},
    {"provider": "xai", "model": "grok-3-fast", "name": "Grok-3-Fast"},
    {"provider": "gemini", "model": "gemini-2.0-flash", "name": "Gemini-2.0-Flash"},
]

SYSTEM_PROMPT = """You are an expert in Power Electronics and Electrical Engineering. 
You are solving GATE (Graduate Aptitude Test in Engineering) Electrical Engineering exam questions.

Your task is to:
1. Carefully read the question about power electronics (converters, inverters, choppers, thyristors, etc.)
2. Apply your knowledge of circuit analysis, semiconductor devices, and converter topologies
3. Select the correct answer from options (A), (B), (C), or (D)

IMPORTANT: 
- Respond with ONLY a single letter: A, B, C, or D
- Do not include any explanation or reasoning
- Do not include parentheses or periods
- Just the letter"""


def extract_answer(response: str) -> Optional[str]:
    """Extract single letter answer from response."""
    response = response.strip().upper()
    
    # Try to find just a single letter A-D
    if response in ['A', 'B', 'C', 'D']:
        return response
    
    # Try to extract from patterns like "(A)" or "A." or "Answer: A"
    patterns = [
        r'^([ABCD])\)?\.?$',
        r'answer[:\s]*([ABCD])',
        r'\(([ABCD])\)',
        r'^([ABCD])[.)]',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    
    # Last resort: find any A-D in the response
    for char in response:
        if char in 'ABCD':
            return char
    
    return None


def query_openai(model: str, question: str) -> Tuple[Optional[str], str]:
    """Query OpenAI model and return answer."""
    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question}
            ],
            max_tokens=10,
            temperature=0.0
        )
        raw_response = response.choices[0].message.content
        answer = extract_answer(raw_response)
        return answer, raw_response
    except Exception as e:
        print(f"  Error with {model}: {e}")
        return None, str(e)


def query_xai(model: str, question: str) -> Tuple[Optional[str], str]:
    """Query xAI Grok model via OpenAI-compatible API."""
    try:
        xai_client = OpenAI(
            api_key=os.getenv("XAI_API_KEY"),
            base_url="https://api.x.ai/v1"
        )
        response = xai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question}
            ],
            max_tokens=10,
            temperature=0.0
        )
        raw_response = response.choices[0].message.content
        answer = extract_answer(raw_response)
        return answer, raw_response
    except Exception as e:
        print(f"  Error with {model}: {e}")
        return None, str(e)


def query_gemini(model: str, question: str) -> Tuple[Optional[str], str]:
    """Query Google Gemini model."""
    if not GEMINI_AVAILABLE:
        return None, "Gemini not available"
    
    try:
        gemini_model = genai.GenerativeModel(model)
        prompt = f"{SYSTEM_PROMPT}\n\nQuestion:\n{question}\n\nAnswer:"
        response = gemini_model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                max_output_tokens=10,
                temperature=0.0
            )
        )
        raw_response = response.text
        answer = extract_answer(raw_response)
        return answer, raw_response
    except Exception as e:
        print(f"  Error with {model}: {e}")
        return None, str(e)


def get_consensus_answer(answers: Dict[str, str]) -> Tuple[Optional[str], float, str]:
    """
    Find consensus answer using majority voting.
    Returns: (answer, confidence, method)
    """
    valid_answers = [a for a in answers.values() if a is not None]
    
    if not valid_answers:
        return None, 0.0, "no_valid_answers"
    
    counter = Counter(valid_answers)
    most_common = counter.most_common()
    
    if len(most_common) == 1:
        # All agree
        return most_common[0][0], 1.0, "unanimous"
    
    top_count = most_common[0][1]
    second_count = most_common[1][1] if len(most_common) > 1 else 0
    
    if top_count > second_count:
        # Clear majority
        confidence = top_count / len(valid_answers)
        return most_common[0][0], confidence, "majority"
    else:
        # Tie - prefer GPT-4o if available
        if answers.get("GPT-4o") in [most_common[0][0], most_common[1][0]]:
            return answers["GPT-4o"], 0.5, "tie_gpt4o"
        return most_common[0][0], 0.5, "tie_first"


def solve_question(question_text: str, question_id: str) -> Dict:
    """Solve a single question using all models."""
    answers = {}
    raw_responses = {}
    
    for model_config in MODELS:
        provider = model_config["provider"]
        model = model_config["model"]
        name = model_config["name"]
        
        if provider == "openai":
            answer, raw = query_openai(model, question_text)
        elif provider == "xai":
            answer, raw = query_xai(model, question_text)
        elif provider == "gemini":
            answer, raw = query_gemini(model, question_text)
        else:
            continue
        
        answers[name] = answer
        raw_responses[name] = raw
        
        # Small delay to avoid rate limits
        time.sleep(0.5)
    
    consensus, confidence, method = get_consensus_answer(answers)
    
    return {
        "question_id": question_id,
        "individual_answers": answers,
        "raw_responses": raw_responses,
        "consensus_answer": consensus,
        "confidence": confidence,
        "consensus_method": method
    }


def is_clean_mcq(problem: Dict) -> bool:
    """Check if a problem is a clean MCQ with options."""
    text = problem.get("problem_text", "")
    
    # Must have all 4 options
    if not all(f'({opt})' in text for opt in ['A', 'B', 'C', 'D']):
        return False
    
    # Not too long (garbled OCR tends to be longer)
    if len(text) > 1500:
        return False
    
    # Avoid page markers at the end
    if 'EE ' in text[-15:]:
        return False
    
    return True


def clean_question_text(text: str) -> str:
    """Clean up OCR artifacts from question text."""
    # Remove page markers
    text = re.sub(r'EE \d+/\d+', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text


def main():
    """Main function to process all GATE questions."""
    # Load GATE problems
    gate_path = Path("benchmarks/expert_verified/gate_official_problems.json")
    
    with open(gate_path) as f:
        gate_data = json.load(f)
    
    # Filter clean MCQ questions
    clean_problems = [p for p in gate_data["problems"] if is_clean_mcq(p)]
    
    print(f"=" * 60)
    print(f"Multi-LLM GATE Answer Generator")
    print(f"=" * 60)
    print(f"Total GATE problems: {len(gate_data['problems'])}")
    print(f"Clean MCQ problems: {len(clean_problems)}")
    print(f"Models: {[m['name'] for m in MODELS]}")
    print(f"=" * 60)
    
    results = []
    stats = {
        "total": len(clean_problems),
        "unanimous": 0,
        "majority": 0,
        "tie": 0,
        "failed": 0
    }
    
    for i, problem in enumerate(clean_problems):
        question_id = problem["id"]
        question_text = clean_question_text(problem["problem_text"])
        
        print(f"\n[{i+1}/{len(clean_problems)}] Processing {question_id}...")
        
        result = solve_question(question_text, question_id)
        results.append(result)
        
        # Update stats
        method = result["consensus_method"]
        if method == "unanimous":
            stats["unanimous"] += 1
        elif method == "majority":
            stats["majority"] += 1
        elif method.startswith("tie"):
            stats["tie"] += 1
        else:
            stats["failed"] += 1
        
        # Print result
        print(f"  Answers: {result['individual_answers']}")
        print(f"  Consensus: {result['consensus_answer']} ({result['confidence']:.0%}, {method})")
    
    # Save results
    output_path = Path("benchmarks/expert_verified/gate_llm_answers.json")
    output_data = {
        "generated_at": datetime.now().isoformat(),
        "models_used": [m["name"] for m in MODELS],
        "statistics": stats,
        "total_questions": len(results),
        "answers": results
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n" + "=" * 60)
    print(f"RESULTS SUMMARY")
    print(f"=" * 60)
    print(f"Total processed: {stats['total']}")
    print(f"Unanimous (100% agreement): {stats['unanimous']}")
    print(f"Majority vote: {stats['majority']}")
    print(f"Ties (50% confidence): {stats['tie']}")
    print(f"Failed: {stats['failed']}")
    print(f"\nResults saved to: {output_path}")
    
    # Also update the original problems with answers
    update_problems_with_answers(gate_data, results)
    
    return results


def update_problems_with_answers(gate_data: Dict, results: List[Dict]):
    """Update GATE problems JSON with consensus answers."""
    # Create lookup
    answer_lookup = {r["question_id"]: r for r in results}
    
    # Update problems
    updated_count = 0
    for problem in gate_data["problems"]:
        if problem["id"] in answer_lookup:
            result = answer_lookup[problem["id"]]
            problem["llm_answer"] = result["consensus_answer"]
            problem["llm_confidence"] = result["confidence"]
            problem["llm_method"] = result["consensus_method"]
            problem["llm_individual"] = result["individual_answers"]
            updated_count += 1
    
    # Save updated file
    output_path = Path("benchmarks/expert_verified/gate_problems_with_answers.json")
    gate_data["answers_generated_at"] = datetime.now().isoformat()
    gate_data["answers_models"] = [m["name"] for m in MODELS]
    
    with open(output_path, 'w') as f:
        json.dump(gate_data, f, indent=2)
    
    print(f"Updated {updated_count} problems with answers")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
