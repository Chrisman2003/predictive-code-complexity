"""
Dataset loader and ground-truth label generator.
Calculates SLOC from code snippets without exposing code to prediction stages.
"""
import re
from typing import Dict, List, Tuple, Union
from src.models.ranking_pipeline.schema import ParadigmImplementation, ProblemItem


def compute_sloc(source_code: str) -> int:
    """
    Computes Source Lines of Code (SLOC) by stripping blank lines
    and C/C++ style single-line and multi-line comments.
    """
    if not source_code:
        return 0
    # Remove multi-line comments /* ... */
    code_no_multi = re.sub(r'/\*.*?\*/', '', source_code, flags=re.DOTALL)
    lines = code_no_multi.splitlines()
    
    sloc_count = 0
    for line in lines:
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
            problem_id = item.get("problem_id") or item.get("id") or "unknown_problem"
            title = item.get("title") or item.get("name") or problem_id
            description = item.get("description", "")
            
            impls = {}
            
            # Option A: Compute SLOC from raw C++ code snippets
            if "implementations" in item and isinstance(item["implementations"], dict):
                for paradigm, code_or_sloc in item["implementations"].items():
                    if isinstance(code_or_sloc, str):
                        sloc = compute_sloc(code_or_sloc)
                        code = code_or_sloc
                    else:
                        sloc = int(code_or_sloc)
                        code = ""
                    impls[paradigm] = ParadigmImplementation(
                        paradigm_name=paradigm,
                        source_code=code,
                        sloc=sloc
                    )
            # Option B: Fallback if pre-computed ground_truth_sloc dictionary is passed
            elif "ground_truth_sloc" in item and isinstance(item["ground_truth_sloc"], dict):
                for paradigm, sloc in item["ground_truth_sloc"].items():
                    impls[paradigm] = ParadigmImplementation(
                        paradigm_name=paradigm,
                        source_code="",
                        sloc=int(sloc)
                    )

            problem = ProblemItem(
                problem_id=problem_id,
                title=title,
                description=description,
                implementations=impls
            )

            # Bind convenience attributes expected by pipeline.py if missing from schema
            if not hasattr(problem, "paradigms"):
                setattr(problem, "paradigms", list(impls.keys()))
            if not hasattr(problem, "ground_truth_sloc"):
                setattr(problem, "ground_truth_sloc", {k: v.sloc for k, v in impls.items()})

            self.problems.append(problem)

    def get_problems(self) -> List[ProblemItem]:
        """Returns all loaded problem items."""
        return self.problems

    def get_ground_truth_ranking(self, problem_id: str) -> List[str]:
        """
        Returns paradigms sorted in descending order of SLOC (most complex -> least complex).
        """
        problem = next(p for p in self.problems if p.problem_id == problem_id)
        sorted_impls = sorted(
            problem.implementations.values(),
            key=lambda x: (x.sloc, x.paradigm_name),
            reverse=True
        )
        return [impl.paradigm_name for impl in sorted_impls]

    def generate_pairwise_combinations(
        self, target: Union[str, List[str]]
    ) -> List[Tuple[str, str]]:
        """
        Generates all pairwise paradigm combinations from either a problem_id or list of paradigm names.
        """
        if isinstance(target, str):
            problem = next(p for p in self.problems if p.problem_id == target)
            paradigms = sorted(list(problem.implementations.keys()))
        else:
            paradigms = sorted(list(target))

        pairs = []
        for i in range(len(paradigms)):
            for j in range(i + 1, len(paradigms)):
                pairs.append((paradigms[i], paradigms[j]))
        return pairs