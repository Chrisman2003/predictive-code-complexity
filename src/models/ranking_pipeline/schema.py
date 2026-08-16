"""
Data schema definitions for the GPU paradigm complexity prediction pipeline.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class ParadigmImplementation:
    paradigm_name: str
    source_code: str
    sloc: Optional[int] = None


@dataclass
class ProblemItem:
    problem_id: str
    title: str
    description: str
    implementations: Dict[str, ParadigmImplementation]  # paradigm_name -> implementation


@dataclass
class PairwisePromptInput:
    problem_id: str
    problem_description: str
    paradigm_a: str
    paradigm_b: str
    few_shot_examples: Optional[List[Dict[str, str]]] = None


@dataclass
class PairwisePrediction:
    problem_id: str
    paradigm_a: str
    paradigm_b: str
    winner: str  # Must be paradigm_a or paradigm_b
    reasoning: str
    raw_response: str


@dataclass
class AggregatedRanking:
    problem_id: str
    predicted_ranking: List[str]  # Ordered from least complex to most complex
    copeland_scores: Dict[str, float]
    win_matrix: Dict[str, Dict[str, int]]
    has_cycles: bool


@dataclass
class EvaluationMetrics:
    problem_id: str
    ground_truth_ranking: List[str]
    predicted_ranking: List[str]
    pairwise_accuracy: float
    kendall_tau: float
    spearman_rho: float
    top1_match: bool