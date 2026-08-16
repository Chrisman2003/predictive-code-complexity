"""
Dataset loader and ground-truth label generator.
Calculates SLOC from code snippets without exposing code to prediction stages.
"""
import re
from typing import Dict, List, Tuple
from models.ranking_pipeline.schema import ParadigmImplementation, ProblemItem


def compute_sloc(source_code: str) -> int:
    """
    Computes Source Lines of Code (SLOC) by stripping blank lines
    and C/C++ style single-line and multi-line comments.
    """
    # Remove multi-line comments /* ... */
    code_no_multi = re.sub(r'/\*.*?\*/', '', source_code, flags=re.DOTALL)
    # Split into lines
    lines = code_no_multi.splitlines()
    
    sloc_count = 0
    for line in lines:
        # Strip single-line comments // ...
        cleaned = re.sub(r'//.*$', '', line).strip()
        if cleaned:
            sloc_count += 1
            
    return sloc_count


class HPCDatasetLoader:
    """
    Loads HPC dataset items, computes ground-truth SLOC labels,
    and isolates natural-language problem descriptions.
    """
    def __init__(self, raw_data: List[Dict]):
        self.problems: List[ProblemItem] = []
        self._parse_raw_data(raw_data)

    def _parse_raw_data(self, raw_data: List[Dict]):
        for item in raw_data:
            problem_id = item["problem_id"]
            title = item["title"]
            description = item["description"]
            
            impls = {}
            for paradigm, code in item["implementations"].items():
                sloc = compute_sloc(code)
                impls[paradigm] = ParadigmImplementation(
                    paradigm_name=paradigm,
                    source_code=code,
                    sloc=sloc
                )
                
            self.problems.append(ProblemItem(
                problem_id=problem_id,
                title=title,
                description=description,
                implementations=impls
            ))

    def get_ground_truth_ranking(self, problem_id: str) -> List[str]:
        """
        Returns paradigms sorted in ascending order of SLOC (least complex -> most complex).
        """
        problem = next(p for p in self.problems if p.problem_id == problem_id)
        sorted_impls = sorted(
            problem.implementations.values(),
            key=lambda x: (x.sloc, x.paradigm_name)
        )
        return [impl.paradigm_name for impl in sorted_impls]

    def generate_pairwise_combinations(self, problem_id: str) -> List[Tuple[str, str]]:
        """
        Generates all K choose 2 paradigm combinations for a given problem.
        """
        problem = next(p for p in self.problems if p.problem_id == problem_id)
        paradigms = sorted(list(problem.implementations.keys()))
        pairs = []
        for i in range(len(paradigms)):
            for j in range(i + 1, len(paradigms)):
                pairs.append((paradigms[i], paradigms[j]))
        return pairs