#!/usr/bin/env python3
"""
LLM Evaluation Framework for PowerElecBench

Evaluates LLM performance on power electronics design problems.
Supports multiple models: GPT-4, Claude, Gemini, local models.

Usage:
    python scripts/evaluate_llm.py --level 1 --num 10 --model gpt-4
    python scripts/evaluate_llm.py --all --model claude-3-opus
"""

import argparse
import json
import os
import sys
import time
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Try to import PySpice for validation
try:
    from PySpice.Spice.NgSpice.Shared import NgSpiceShared
    PYSPICE_AVAILABLE = True
except ImportError:
    PYSPICE_AVAILABLE = False
    print("⚠️  PySpice not available - simulation validation disabled")


@dataclass
class EvalResult:
    """Result of evaluating a single problem"""
    problem_id: str
    level: int
    topology: str
    prompt: str
    
    # Ground truth
    gt_vout: float
    gt_components: Dict
    
    # LLM response
    llm_response: str
    llm_vout: Optional[float] = None
    llm_components: Optional[Dict] = None
    
    # Metrics
    parse_success: bool = False
    vout_error_pct: Optional[float] = None
    component_match_score: Optional[float] = None
    simulation_success: bool = False
    sim_vout: Optional[float] = None
    sim_error_pct: Optional[float] = None
    
    # Timing
    latency_ms: float = 0
    tokens_in: int = 0
    tokens_out: int = 0
    
    error_msg: Optional[str] = None


class BenchmarkLoader:
    """Loads benchmark problems from JSON files"""
    
    def __init__(self, benchmark_dir: Path):
        self.benchmark_dir = benchmark_dir
        self.manifest = self._load_manifest()
        
    def _load_manifest(self) -> Dict:
        manifest_path = self.benchmark_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                return json.load(f)
        return {}
    
    def load_level(self, level: int) -> List[Dict]:
        """Load all problems for a given level"""
        problems = []
        level_dir = self.benchmark_dir / f"level_{level}"
        
        if not level_dir.exists():
            print(f"⚠️  Level {level} directory not found")
            return []
        
        # Load from problems_*.json files (they have specs.vout for ground truth)
        for prob_file in sorted(level_dir.glob("problems_*.json")):
            with open(prob_file) as f:
                data = json.load(f)
                if "problems" in data:
                    for prob in data["problems"]:
                        problem_id = prob.get("id", "unknown")
                        prompt = prob.get("prompt", "")
                        specs = prob.get("specs", {})
                        
                        # Build solution dict from specs
                        solution = {
                            "topology": specs.get("topology") or prob.get("topology"),
                            "vout": specs.get("vout"),
                            "vin": specs.get("vin"),
                            "iout": specs.get("iout"),
                            "power": specs.get("power"),
                            "f_sw": specs.get("f_sw"),
                        }
                        
                        problems.append({
                            "id": problem_id,
                            "level": level,
                            "prompt": prompt,
                            "solution": solution,
                        })
        
        # Fallback: also check solution files if no problems found
        if not problems:
            for sol_file in sorted(level_dir.glob("solutions_*.json")):
                with open(sol_file) as f:
                    data = json.load(f)
                if "solutions" in data:
                    for sol in data["solutions"]:
                        # Handle different formats
                        problem_id = sol.get("id") or sol.get("problem_id", "unknown")
                        
                        # Build prompt from either direct prompt or specifications
                        if sol.get("prompt"):
                            prompt = sol["prompt"]
                        elif sol.get("problem"):
                            prompt = sol["problem"]
                        elif sol.get("specifications"):
                            # Build prompt from specifications
                            specs = sol["specifications"]
                            title = sol.get("title", "")
                            topo = sol.get("topology", "converter")
                            prompt = f"Design a {topo}: "
                            if specs.get("input_voltage"):
                                prompt += f"{specs['input_voltage']}V input to "
                            if specs.get("output_voltage"):
                                prompt += f"{specs['output_voltage']}V output"
                            if specs.get("output_current"):
                                prompt += f", {specs['output_current']}A load"
                            elif specs.get("output_power"):
                                prompt += f", {specs['output_power']}W"
                        else:
                            prompt = ""
                        
                        # Get solution data
                        solution = sol.get("solution", {})
                        if not solution and sol.get("specifications"):
                            # Build solution from specifications
                            specs = sol["specifications"]
                            solution = {
                                "topology": sol.get("topology"),
                                "vout": specs.get("output_voltage"),
                                "vin": specs.get("input_voltage"),
                                "components": sol.get("component_ratings", {}),
                            }
                        
                        problems.append({
                            "id": problem_id,
                            "level": level,
                            "prompt": prompt,
                            "solution": solution,
                        })
        
        return problems
    
    def load_all(self) -> Dict[int, List[Dict]]:
        """Load all problems organized by level"""
        all_problems = {}
        for level in [1, 2, 3, 4]:
            problems = self.load_level(level)
            if problems:
                all_problems[level] = problems
        return all_problems


class LLMClient:
    """Unified interface for different LLM providers"""
    
    def __init__(self, model: str, api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key
        
        # Determine provider and set appropriate key
        if "gpt" in model.lower() or "o1" in model.lower() or model.startswith("ft:"):
            self.provider = "openai"
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        elif "claude" in model.lower():
            self.provider = "anthropic"
            self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        elif "gemini" in model.lower():
            self.provider = "google"
            self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        elif "grok" in model.lower():
            self.provider = "xai"
            self.api_key = api_key or os.getenv("XAI_API_KEY")
        elif "llama" in model.lower() or "mixtral" in model.lower():
            # Use Groq for LLaMA (fast, free tier available)
            self.provider = "groq"
            self.api_key = api_key or os.getenv("GROQ_API_KEY")
        elif "together" in model.lower():
            # Together.ai for various open models
            self.provider = "together"
            self.api_key = api_key or os.getenv("TOGETHER_API_KEY")
        else:
            self.provider = "openai"
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
    
    def generate(self, prompt: str, system_prompt: str = "") -> Tuple[str, Dict]:
        """Generate response from LLM. Returns (response, metadata)"""
        
        if self.provider == "openai":
            return self._call_openai(prompt, system_prompt)
        elif self.provider == "anthropic":
            return self._call_anthropic(prompt, system_prompt)
        elif self.provider == "google":
            return self._call_gemini(prompt, system_prompt)
        elif self.provider == "xai":
            return self._call_xai(prompt, system_prompt)
        elif self.provider == "groq":
            return self._call_groq(prompt, system_prompt)
        elif self.provider == "together":
            return self._call_together(prompt, system_prompt)
        else:
            return self._call_openai(prompt, system_prompt)
    
    def _call_xai(self, prompt: str, system_prompt: str) -> Tuple[str, Dict]:
        """Call xAI Grok API (OpenAI-compatible)"""
        try:
            from openai import OpenAI
            
            client = OpenAI(
                api_key=self.api_key,
                base_url="https://api.x.ai/v1"
            )
            
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            start = time.time()
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,
                max_tokens=2000,
            )
            latency = (time.time() - start) * 1000
            
            return response.choices[0].message.content, {
                "latency_ms": latency,
                "tokens_in": response.usage.prompt_tokens,
                "tokens_out": response.usage.completion_tokens,
            }
        except Exception as e:
            return f"ERROR: {e}", {"latency_ms": 0, "tokens_in": 0, "tokens_out": 0}
    
    def _call_openai(self, prompt: str, system_prompt: str) -> Tuple[str, Dict]:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
            
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            start = time.time()
            
            # GPT-5 mini/nano don't support temperature, use default
            # GPT-5.1 and older models support temperature
            is_gpt5_restricted = self.model in ["gpt-5-mini", "gpt-5-nano"]
            is_new_api = self.model.startswith("gpt-5") or self.model.startswith("o1") or self.model.startswith("o3")
            
            if is_gpt5_restricted:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_completion_tokens=2000,
                )
            elif is_new_api:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.1,
                    max_completion_tokens=2000,
                )
            else:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=2000,
                )
            latency = (time.time() - start) * 1000
            
            return response.choices[0].message.content, {
                "latency_ms": latency,
                "tokens_in": response.usage.prompt_tokens,
                "tokens_out": response.usage.completion_tokens,
            }
        except Exception as e:
            return f"ERROR: {e}", {"latency_ms": 0, "tokens_in": 0, "tokens_out": 0}
    
    def _call_anthropic(self, prompt: str, system_prompt: str) -> Tuple[str, Dict]:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)
            
            start = time.time()
            response = client.messages.create(
                model=self.model,
                max_tokens=2000,
                system=system_prompt if system_prompt else "You are a power electronics design expert.",
                messages=[{"role": "user", "content": prompt}],
            )
            latency = (time.time() - start) * 1000
            
            return response.content[0].text, {
                "latency_ms": latency,
                "tokens_in": response.usage.input_tokens,
                "tokens_out": response.usage.output_tokens,
            }
        except Exception as e:
            return f"ERROR: {e}", {"latency_ms": 0, "tokens_in": 0, "tokens_out": 0}
    
    def _call_gemini(self, prompt: str, system_prompt: str) -> Tuple[str, Dict]:
        try:
            import google.generativeai as genai
            from google.generativeai.types import HarmCategory, HarmBlockThreshold
            
            genai.configure(api_key=self.api_key)
            
            # Create model with system instruction
            model = genai.GenerativeModel(
                model_name=self.model,
                system_instruction=system_prompt if system_prompt else "You are a power electronics design expert."
            )
            
            # Disable safety filters for technical content
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
            
            start = time.time()
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=2000,
                ),
                safety_settings=safety_settings,
            )
            latency = (time.time() - start) * 1000
            
            # Extract token counts if available
            tokens_in = 0
            tokens_out = 0
            if hasattr(response, 'usage_metadata'):
                tokens_in = getattr(response.usage_metadata, 'prompt_token_count', 0)
                tokens_out = getattr(response.usage_metadata, 'candidates_token_count', 0)
            
            # Handle blocked responses
            if not response.candidates or not response.candidates[0].content.parts:
                return f"ERROR: Response blocked (finish_reason: {response.candidates[0].finish_reason if response.candidates else 'unknown'})", {
                    "latency_ms": latency, "tokens_in": tokens_in, "tokens_out": tokens_out
                }
            
            return response.text, {
                "latency_ms": latency,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
            }
        except Exception as e:
            return f"ERROR: {e}", {"latency_ms": 0, "tokens_in": 0, "tokens_out": 0}

    def _call_groq(self, prompt: str, system_prompt: str) -> Tuple[str, Dict]:
        """Call Groq API for LLaMA and Mixtral models (OpenAI-compatible)"""
        try:
            from openai import OpenAI
            
            client = OpenAI(
                api_key=self.api_key,
                base_url="https://api.groq.com/openai/v1"
            )
            
            # Map model names to Groq model IDs
            model_map = {
                "llama-3.1-8b": "llama-3.1-8b-instant",
                "llama-3.1-70b": "llama-3.1-70b-versatile",
                "llama-3.2-3b": "llama-3.2-3b-preview",
                "llama-3.2-90b": "llama-3.2-90b-vision-preview",
                "llama-3.3-70b": "llama-3.3-70b-versatile",
                "mixtral-8x7b": "mixtral-8x7b-32768",
            }
            groq_model = model_map.get(self.model.lower(), self.model)
            
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            start = time.time()
            response = client.chat.completions.create(
                model=groq_model,
                messages=messages,
                temperature=0.1,
                max_tokens=2000,
            )
            latency = (time.time() - start) * 1000
            
            return response.choices[0].message.content, {
                "latency_ms": latency,
                "tokens_in": response.usage.prompt_tokens,
                "tokens_out": response.usage.completion_tokens,
            }
        except Exception as e:
            return f"ERROR: {e}", {"latency_ms": 0, "tokens_in": 0, "tokens_out": 0}

    def _call_together(self, prompt: str, system_prompt: str) -> Tuple[str, Dict]:
        """Call Together.ai API (OpenAI-compatible)"""
        try:
            from openai import OpenAI
            
            client = OpenAI(
                api_key=self.api_key,
                base_url="https://api.together.xyz/v1"
            )
            
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            start = time.time()
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,
                max_tokens=2000,
            )
            latency = (time.time() - start) * 1000
            
            return response.choices[0].message.content, {
                "latency_ms": latency,
                "tokens_in": response.usage.prompt_tokens,
                "tokens_out": response.usage.completion_tokens,
            }
        except Exception as e:
            return f"ERROR: {e}", {"latency_ms": 0, "tokens_in": 0, "tokens_out": 0}


class ResponseParser:
    """Parse LLM responses to extract design parameters"""
    
    @staticmethod
    def _clean_number(s: str) -> float:
        """Clean and convert a numeric string, handling trailing dots etc."""
        s = s.strip().rstrip('.')
        return float(s) if s else 0.0
    
    @staticmethod
    def parse_design(response: str) -> Dict:
        """Extract circuit parameters from LLM response"""
        result = {
            "vout": None,
            "duty_cycle": None,
            "L": None,
            "C_out": None,
            "C_in": None,
            "f_sw": None,
            "topology": None,
        }
        
        # Extract topology
        topo_match = re.search(r"(buck|boost|sepic|cuk|flyback|forward|half[_-]?bridge|full[_-]?bridge)", 
                               response.lower())
        if topo_match:
            result["topology"] = topo_match.group(1).replace("-", "_").replace(" ", "_")
        
        # Extract duty cycle - multiple formats
        duty_patterns = [
            r"\*\*[Dd]uty\s*[Cc]ycle[:\*\s]+([0-9.]+)",
            r"[Dd]uty\s*[Cc]ycle[:\s=]+([0-9.]+)",
            r"D\s*=\s*([0-9.]+)",
            r"duty[:\s=]+([0-9.]+)",
        ]
        for pattern in duty_patterns:
            match = re.search(pattern, response)
            if match:
                try:
                    val = ResponseParser._clean_number(match.group(1))
                    result["duty_cycle"] = val if val <= 1 else val / 100
                    break
                except:
                    pass
        
        # Extract output voltage - multiple formats (order matters - most specific first)
        vout_patterns = [
            r"output\s+voltage\s+\(([0-9.]+)\s*V(?:\)|[^)]*\))",  # output voltage (5V) or (5V for USB)
            r"to\s+a\s+(?:higher|lower)\s+output\s+voltage\s+\(([0-9.]+)\s*V",  # to a lower output voltage (5V
            r"from\s+[0-9.]+\s*V\s+to\s+([0-9.]+)\s*V",  # from 24V to 12V
            r"[Oo]utput[:\s]+([0-9.]+)\s*V(?:\s|,|$)",  # Output: 5V or Output 5V
            r"V_?out\s*[=:]\s*([0-9.]+)\s*V",  # Vout = 5V or V_out: 5V
            r"\*\*[Oo]utput\s*[Vv]oltage[:\*\s]*([0-9.]+)\s*V",  # **Output Voltage:** 5V
            r"\*\*V_?out[:\*\s]*([0-9.]+)\s*V",  # **Vout:** 5V
            r"[Oo]utput\s*[Vv]oltage[:\s]+([0-9.]+)\s*V",  # Output Voltage: 5V
            r"[Ee]xpected\s*[Vv]_?out[:\s=]+([0-9.]+)\s*V",
            r"[Ff]inal\s+[Oo]utput[:\s]+([0-9.]+)\s*V",  # Final Output: 5V
            r"[Dd]esired\s+[Oo]utput[:\s]+([0-9.]+)\s*V",  # Desired Output: 5V
            r"[Tt]arget\s+[Vv]oltage[:\s]+([0-9.]+)\s*V",  # Target Voltage: 5V
            r"([0-9.]+)\s*V\s+(?:DC\s+)?[Oo]utput",  # 5V output or 5V DC output
            r"[Oo]utput\s+of\s+([0-9.]+)\s*V",  # output of 5V
            r"delivers?\s+[0-9.]+\s*A?\s*(?:at\s+)?([0-9.]+)\s*V",  # delivers 2A at 5V
            r"([0-9.]+)\s*V\s*\([0-9]+\s*W\)",  # 5V (10W)
            r"produces?\s+([0-9.]+)\s*V",  # produces 5V
            r"[Rr]egulated\s+to\s+([0-9.]+)\s*V",  # Regulated to 5V
            r"step\s+down\s+to\s+([0-9.]+)\s*V",  # step down to 48V
            r"([0-9.]+)\s*V[/\\]([0-9.]+)\s*A",  # 48V/20A format - captures voltage
            r"→\s*([0-9.]+)\s*V",  # → 48V
            r"->\s*([0-9.]+)\s*V",  # -> 48V
            r"to\s+([0-9.]+)\s*V\s+DC",  # to 48V DC
        ]
        
        # First, try to find target/final/output voltage with strong indicators
        strong_indicators = [
            r"[Rr]egulated\s+to\s+([0-9.]+)\s*V",  # Regulated to 5V
            r"\*\*[Ee]xpected\s+[Oo]utput\s+[Vv]oltage[^*]*\*\*[:\s]*([0-9.]+)\s*V",  # **Expected Output Voltage (V_out)**: 400 V
            r"\*\*[Ee]xpected\s+V_?out\*?\*?\s*[:\s]\s*([0-9.]+)\s*V",  # **Expected Vout**: 3.3 V
            r"[Ee]xpected\s+V_?out\s*[:\s]\s*([0-9.]+)\s*V",  # Expected Vout: 3.3 V
            r"[Ee]xpected\s+[Oo]utput[:\s]+\\?\(?V_?\{?out\}?\s*=\s*([0-9.]+)",  # Expected output: V_{out} = 5.0
            r"final\s+output[:\s]*([0-9.]+)\s*V",  # final output: 48V
            r"output\s+voltage[:\s]*([0-9.]+)\s*V",  # output voltage: 48V
            r"target\s+output[:\s]*([0-9.]+)\s*V",  # target output: 48V
            r"([0-9.]+)\s*V\s+(?:regulated\s+)?output",  # 48V output
            r"(\d+)\s*V[/\\]\d+\s*A\s+(?:server|PSU|supply)",  # 48V/20A server
            r"delivers?\s+\d+\s*A?\s*(?:at\s+)?(\d+)\s*V",  # delivers 2A at 48V
            r"step.*down.*to\s+(\d+)\s*V",  # step down to 48V
            r"→\s*(\d+)\s*V(?:\s*DC)?(?:\s|,|$)",  # → 48V DC
            r"->\s*(\d+)\s*V(?:\s*DC)?(?:\s|,|$)",  # -> 48V DC
            r"(?:secondary|second)\s+stage.*?(\d+)\s*V",  # second stage ... 48V
        ]
        
        for pattern in strong_indicators:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                try:
                    val = ResponseParser._clean_number(match.group(1))
                    # Skip intermediate bus voltages (typically 380-420V for PFC)
                    if 350 < val < 450:
                        continue
                    result["vout"] = val
                    break
                except:
                    pass
        
        # If no strong match, try general patterns
        if result["vout"] is None:
            for pattern in vout_patterns:
                match = re.search(pattern, response)
                if match:
                    try:
                        val = ResponseParser._clean_number(match.group(1))
                        # Skip intermediate bus voltages
                        if 350 < val < 450:
                            continue
                        result["vout"] = val
                        break
                    except:
                        pass
        
        # Extract inductor value - multiple formats
        L_patterns = [
            r"\*\*[Ii]nductor[:\*\s]+([0-9.]+)\s*[uμµ]H",
            r"\*\*[Ii]nductor[:\*\s]+([0-9.]+)\s*mH",
            r"[Ll](?:inductor)?[:\s=]+([0-9.]+)\s*[uμµ]H",
            r"([0-9.]+)\s*[uμµ]H\s*inductor",
            r"[Ii]nductor[:\s=]+([0-9.]+)\s*[uμµ]H",
            r"L\s*=\s*([0-9.]+)\s*[uμµ]H",
        ]
        for pattern in L_patterns:
            match = re.search(pattern, response)
            if match:
                try:
                    val = ResponseParser._clean_number(match.group(1))
                    # Check if it was in mH format
                    if "mH" in pattern:
                        result["L"] = val * 1e-3
                    else:
                        result["L"] = val * 1e-6
                    break
                except:
                    pass
        
        # Extract output capacitor - multiple formats
        Cout_patterns = [
            r"\*\*[Oo]utput\s*[Cc]apacitor[:\*\s]+([0-9.]+)\s*[uμµ]F",
            r"[Cc]_?out[:\s=]+([0-9.]+)\s*[uμµ]F",
            r"[Oo]utput\s+[Cc]apacitor[:\s=]+([0-9.]+)\s*[uμµ]F",
            r"([0-9.]+)\s*[uμµ]F\s*output",
            r"C\s*=\s*([0-9.]+)\s*[uμµ]F",
        ]
        for pattern in Cout_patterns:
            match = re.search(pattern, response)
            if match:
                try:
                    result["C_out"] = ResponseParser._clean_number(match.group(1)) * 1e-6
                    break
                except:
                    pass
        
        # Extract switching frequency - multiple formats
        fsw_patterns = [
            r"\*\*[Ss]witching\s*[Ff]requency[:\*\s]+([0-9.]+)\s*kHz",
            r"[Ff]_?sw[:\s=]+([0-9.]+)\s*kHz",
            r"[Ss]witching\s+[Ff]requency[:\s=]+([0-9.]+)\s*kHz",
            r"([0-9.]+)\s*kHz\s*switching",
            r"f_{?sw}?\s*=\s*([0-9.]+)\s*kHz",
        ]
        for pattern in fsw_patterns:
            match = re.search(pattern, response)
            if match:
                try:
                    result["f_sw"] = ResponseParser._clean_number(match.group(1)) * 1e3
                    break
                except:
                    pass
        
        return result


class Evaluator:
    """Main evaluation orchestrator"""
    
    SYSTEM_PROMPT = """You are an expert power electronics engineer. 
When given a converter design problem, provide:
1. The topology choice and justification
2. Key component values (inductor, capacitors) with calculations
3. Duty cycle calculation
4. Expected output voltage and ripple

Format your response clearly with labeled values like:
- Topology: buck
- Duty Cycle: 0.417
- Inductor: 47µH
- Output Capacitor: 100µF
- Switching Frequency: 200kHz
- Expected Vout: 5.0V
"""
    
    def __init__(self, model: str, api_key: Optional[str] = None):
        self.llm = LLMClient(model, api_key)
        self.parser = ResponseParser()
        self.loader = BenchmarkLoader(PROJECT_ROOT / "benchmarks")
        self.results: List[EvalResult] = []
    
    def evaluate_problem(self, problem: Dict) -> EvalResult:
        """Evaluate a single problem"""
        solution = problem.get("solution", {})
        
        result = EvalResult(
            problem_id=problem["id"],
            level=problem["level"],
            topology=solution.get("topology", "unknown"),
            prompt=problem["prompt"],
            gt_vout=solution.get("vout", 0),
            gt_components=solution.get("components", {}),
            llm_response="",
        )
        
        # Call LLM
        try:
            response, metadata = self.llm.generate(problem["prompt"], self.SYSTEM_PROMPT)
            result.llm_response = response
            result.latency_ms = metadata.get("latency_ms", 0)
            result.tokens_in = metadata.get("tokens_in", 0)
            result.tokens_out = metadata.get("tokens_out", 0)
            
            if "ERROR:" in response:
                result.error_msg = response
                return result
            
            # Parse response
            parsed = self.parser.parse_design(response)
            result.llm_vout = parsed.get("vout")
            result.llm_components = parsed
            result.parse_success = any(v is not None for v in parsed.values())
            
            # Calculate metrics
            if result.llm_vout and result.gt_vout:
                result.vout_error_pct = abs(result.llm_vout - result.gt_vout) / abs(result.gt_vout) * 100
            
            # Component matching score
            if result.llm_components and result.gt_components:
                result.component_match_score = self._calc_component_score(
                    result.llm_components, result.gt_components
                )
            
        except Exception as e:
            result.error_msg = str(e)
        
        return result
    
    def _calc_component_score(self, llm_comp: Dict, gt_comp: Dict) -> float:
        """Calculate component matching score (0-100)"""
        scores = []
        
        # Check inductor
        if "L" in llm_comp and llm_comp["L"] and "L" in gt_comp:
            gt_L = gt_comp["L"].get("value", 0) if isinstance(gt_comp["L"], dict) else gt_comp["L"]
            if gt_L > 0:
                error = abs(llm_comp["L"] - gt_L) / gt_L
                scores.append(max(0, 100 - error * 100))
        
        # Check capacitor
        if "C_out" in llm_comp and llm_comp["C_out"] and "C_out" in gt_comp:
            gt_C = gt_comp["C_out"].get("value", 0) if isinstance(gt_comp["C_out"], dict) else gt_comp["C_out"]
            if gt_C > 0:
                error = abs(llm_comp["C_out"] - gt_C) / gt_C
                scores.append(max(0, 100 - error * 100))
        
        return sum(scores) / len(scores) if scores else 0
    
    def evaluate_level(self, level: int, num_problems: Optional[int] = None) -> List[EvalResult]:
        """Evaluate all problems in a level"""
        problems = self.loader.load_level(level)
        
        if num_problems:
            problems = problems[:num_problems]
        
        print(f"\n{'='*60}")
        print(f"Evaluating Level {level}: {len(problems)} problems")
        print(f"{'='*60}")
        
        results = []
        for i, problem in enumerate(problems):
            print(f"\n[{i+1}/{len(problems)}] {problem['id']}: ", end="", flush=True)
            
            result = self.evaluate_problem(problem)
            results.append(result)
            self.results.append(result)
            
            if result.parse_success:
                vout_str = f"Vout={result.llm_vout:.1f}V" if result.llm_vout else "Vout=?"
                err_str = f"err={result.vout_error_pct:.1f}%" if result.vout_error_pct is not None else ""
                print(f"✅ {vout_str} {err_str}")
            else:
                print(f"❌ Parse failed: {result.error_msg or 'Unknown'}")
            
            # Rate limiting
            time.sleep(0.5)
        
        return results
    
    def print_summary(self):
        """Print evaluation summary"""
        print(f"\n{'='*70}")
        print("EVALUATION SUMMARY")
        print(f"{'='*70}")
        
        # By level
        by_level = defaultdict(list)
        for r in self.results:
            by_level[r.level].append(r)
        
        print(f"\n{'Level':<8} {'Total':<8} {'Parsed':<10} {'Vout<5%':<10} {'Vout<10%':<10} {'Avg Err':<10}")
        print("-" * 60)
        
        total_parsed = 0
        total_under5 = 0
        total_under10 = 0
        total_count = 0
        all_errors = []
        
        for level in sorted(by_level.keys()):
            results = by_level[level]
            parsed = sum(1 for r in results if r.parse_success)
            under5 = sum(1 for r in results if r.vout_error_pct is not None and r.vout_error_pct < 5)
            under10 = sum(1 for r in results if r.vout_error_pct is not None and r.vout_error_pct < 10)
            errors = [r.vout_error_pct for r in results if r.vout_error_pct is not None]
            avg_err = sum(errors) / len(errors) if errors else float('inf')
            
            total_parsed += parsed
            total_under5 += under5
            total_under10 += under10
            total_count += len(results)
            all_errors.extend(errors)
            
            print(f"L{level:<7} {len(results):<8} {parsed:<10} {under5:<10} {under10:<10} {avg_err:.1f}%")
        
        print("-" * 60)
        avg_all = sum(all_errors) / len(all_errors) if all_errors else float('inf')
        print(f"{'TOTAL':<8} {total_count:<8} {total_parsed:<10} {total_under5:<10} {total_under10:<10} {avg_all:.1f}%")
        
        # Latency stats
        latencies = [r.latency_ms for r in self.results if r.latency_ms > 0]
        if latencies:
            print(f"\nLatency: avg={sum(latencies)/len(latencies):.0f}ms, "
                  f"min={min(latencies):.0f}ms, max={max(latencies):.0f}ms")
        
        # Token stats
        tokens_in = sum(r.tokens_in for r in self.results)
        tokens_out = sum(r.tokens_out for r in self.results)
        print(f"Tokens: {tokens_in:,} in, {tokens_out:,} out, {tokens_in+tokens_out:,} total")
    
    def save_results(self, output_path: Path):
        """Save results to JSON"""
        output = {
            "timestamp": datetime.now().isoformat(),
            "model": self.llm.model,
            "total_problems": len(self.results),
            "results": [asdict(r) for r in self.results],
        }
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        print(f"\n📁 Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="PowerElecBench LLM Evaluator")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                       help="Model to evaluate (gpt-4, gpt-4o, claude-3-opus, etc.)")
    parser.add_argument("--level", type=int, default=None,
                       help="Specific level to evaluate (1-4)")
    parser.add_argument("--all", action="store_true",
                       help="Evaluate all levels")
    parser.add_argument("--num", type=int, default=10,
                       help="Number of problems per level (default: 10)")
    parser.add_argument("--api_key", type=str, default=None,
                       help="API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output file for results")
    
    args = parser.parse_args()
    
    print(f"\n🔬 PowerElecBench Evaluation")
    print(f"   Model: {args.model}")
    print(f"   Problems per level: {args.num}")
    
    evaluator = Evaluator(args.model, args.api_key)
    
    if args.all:
        for level in [1, 2, 3, 4]:
            evaluator.evaluate_level(level, args.num)
    elif args.level:
        evaluator.evaluate_level(args.level, args.num)
    else:
        # Default: evaluate level 1
        evaluator.evaluate_level(1, args.num)
    
    evaluator.print_summary()
    
    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = PROJECT_ROOT / "benchmarks" / "results" / f"eval_{args.model}_{timestamp}.json"
        output_path.parent.mkdir(exist_ok=True)
    
    evaluator.save_results(output_path)


if __name__ == "__main__":
    main()
