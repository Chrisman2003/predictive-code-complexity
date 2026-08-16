"""
Context Builder Subpipeline.
Compiles Global, Local, and Historical context into a structured Knowledge Base
to prime the LLM before pairwise inference.
"""
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class GlobalContext:
    """Information about the programming paradigms themselves."""
    paradigm_descriptions: Dict[str, str] = field(default_factory=dict)
    memory_models: Dict[str, str] = field(default_factory=dict)


@dataclass
class LocalContext:
    """Information about the specific repository or benchmark suite."""
    repository_name: str = ""
    repository_goals: str = ""
    architectural_constraints: str = ""


@dataclass
class HistoricalContext:
    """Information regarding ecosystem maturity, integration costs, and toolchain overhead."""
    ecosystem_maturity: Dict[str, str] = field(default_factory=dict)
    known_frictions: Dict[str, str] = field(default_factory=dict)


class ContextManager:
    """
    Manages and compiles distinct knowledge domains into a unified context preamble
    for the LLM.
    """
    def __init__(self):
        self.global_ctx = GlobalContext()
        self.local_ctx = LocalContext()
        self.historical_ctx = HistoricalContext()

    def load_from_dictionaries(self, global_data: dict, local_data: dict, historical_data: dict):
        """Loads context data from standard dictionaries (easily adaptable to JSON)."""
        self.global_ctx.paradigm_descriptions = global_data.get("descriptions", {})
        self.global_ctx.memory_models = global_data.get("memory_models", {})
        
        self.local_ctx.repository_name = local_data.get("repository_name", "")
        self.local_ctx.repository_goals = local_data.get("repository_goals", "")
        self.local_ctx.architectural_constraints = local_data.get("architectural_constraints", "")
        
        self.historical_ctx.ecosystem_maturity = historical_data.get("maturity", {})
        self.historical_ctx.known_frictions = historical_data.get("known_frictions", {})

    def build_context_preamble(self, paradigm_a: str, paradigm_b: str) -> str:
        """
        Compiles a targeted knowledge base string containing ONLY the context 
        relevant to the repository and the two paradigms currently being compared.
        """
        parts = []
        parts.append("============================================================")
        parts.append("PROVIDED KNOWLEDGE BASE (CONTEXT)")
        parts.append("============================================================")

        # 1. Local Context (Always included)
        if self.local_ctx.repository_name:
            parts.append("\n[LOCAL REPOSITORY CONTEXT]")
            parts.append(f"Repository: {self.local_ctx.repository_name}")
            parts.append(f"Goals: {self.local_ctx.repository_goals}")
            parts.append(f"Constraints: {self.local_ctx.architectural_constraints}")

        # 2. Global Context (Filtered for the current pair)
        parts.append("\n[GLOBAL PARADIGM CONTEXT]")
        for p in [paradigm_a, paradigm_b]:
            desc = self.global_ctx.paradigm_descriptions.get(p, "No general description provided.")
            mem = self.global_ctx.memory_models.get(p, "No memory model provided.")
            parts.append(f"- {p} Overview: {desc}")
            parts.append(f"- {p} Memory Model: {mem}")

        # 3. Historical Context (Filtered for the current pair)
        parts.append("\n[HISTORICAL & ECOSYSTEM CONTEXT]")
        parts.append("Note: Unfamiliar, highly verbose, or poorly integrated paradigms inherently increase implementation complexity (SLOC).")
        for p in [paradigm_a, paradigm_b]:
            mat = self.historical_ctx.ecosystem_maturity.get(p, "Unknown maturity.")
            fric = self.historical_ctx.known_frictions.get(p, "No known frictions.")
            parts.append(f"- {p} Maturity: {mat}")
            parts.append(f"- {p} Known Frictions: {fric}")

        parts.append("============================================================\n")
        return "\n".join(parts)