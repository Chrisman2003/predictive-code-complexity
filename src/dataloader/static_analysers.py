import math
from typing import Dict, Any
import lizard

class CodeAnalyzer:
    """Extracts SLOC, Cyclomatic Complexity, and Halstead Metrics from source files."""

    @staticmethod
    def analyze_file(file_path: str) -> Dict[str, Any]:
        analysis = lizard.analyze_file(file_path)
        
        # Aggregate function-level metrics
        total_nloc = analysis.nloc
        function_count = len(analysis.function_list)
        tot_cyclomatic = sum(f.cyclomatic_complexity for f in analysis.function_list)
        total_tokens = sum(f.token_count for f in analysis.function_list)

        # NOTE: Only fundamental metrics are to be predicted. Derived metrics may be directly computed from these.
        return {
            "sloc": total_nloc,
            "function_count": function_count,
            "tot_cyclomatic": tot_cyclomatic,
            "tokens" : total_tokens,
        }