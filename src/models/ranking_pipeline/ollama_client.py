"""
Robust Client for interacting with local Ollama HTTP endpoint.
Features JSON response parsing, validation, and retry logic.
"""
import json
import re
import time
import urllib.error
import urllib.request
from typing import Dict, Optional
from models.ranking_pipeline.prompts import PromptFactory

class OllamaClient:
    """
    Communicates with Ollama server running locally (default: http://localhost:11434).
    """

    def __init__(
        self,
        model_name: str = "qwen2.5-coder",
        host: str = "http://localhost:11434",
        timeout: int = 60,
        max_retries: int = 3
    ):
        self.model_name = model_name
        self.host = host.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries

    def predict_pairwise(
        self,
        problem_description: str,
        paradigm_a: str,
        paradigm_b: str,
        few_shot_examples: Optional[list] = None
    ) -> Dict[str, str]:
        """
        Queries Ollama for a pairwise comparison between paradigm_a and paradigm_b.
        Guarantees structured output containing 'winner' and 'reasoning'.
        """
        system_prompt = PromptFactory.get_system_prompt()
        user_prompt = PromptFactory.build_pairwise_prompt(
            problem_description=problem_description,
            paradigm_a=paradigm_a,
            paradigm_b=paradigm_b,
            few_shot_examples=few_shot_examples
        )

        for attempt in range(1, self.max_retries + 1):
            try:
                raw_response = self._call_generate_api(system_prompt, user_prompt)
                parsed = self._extract_json(raw_response)

                winner = parsed.get("winner", "").strip()
                reasoning = parsed.get("reasoning", "").strip()

                # Validate winner value
                if winner not in [paradigm_a, paradigm_b]:
                    # Fuzzy correction if LLM slightly altered casing or spacing
                    if paradigm_a.lower() in winner.lower():
                        winner = paradigm_a
                    elif paradigm_b.lower() in winner.lower():
                        winner = paradigm_b
                    else:
                        raise ValueError(
                            f"Invalid winner '{winner}'. Expected '{paradigm_a}' or '{paradigm_b}'."
                        )

                return {
                    "winner": winner,
                    "reasoning": reasoning,
                    "raw_response": raw_response
                }

            except Exception as e:
                if attempt == self.max_retries:
                    # Deterministic fallback on final failure to prevent pipeline crash
                    print(f"[WARNING] Failed after {self.max_retries} attempts: {e}. Falling back to default.")
                    return {
                        "winner": paradigm_a,
                        "reasoning": f"Fallback due to parsing error: {e}",
                        "raw_response": ""
                    }
                time.sleep(1.0 * attempt)

    def _call_generate_api(self, system_prompt: str, user_prompt: str) -> str:
        url = f"{self.host}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": f"{system_prompt}\n\n{user_prompt}",
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.0,  # Zero-temperature for maximum determinism
                "top_p": 0.1
            }
        }

        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(
            url,
            data=data,
            headers={'Content-Type': 'application/json'}
        )

        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            res_body = response.read().decode('utf-8')
            res_json = json.loads(res_body)
            return res_json.get("response", "")

    def _extract_json(self, text: str) -> Dict:
        """
        Parses JSON object from raw response string.
        Handles wrapped code blocks ```json ... ``` gracefully.
        """
        text = text.strip()
        # Regex to find JSON block
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            return json.loads(json_str)
        return json.loads(text)