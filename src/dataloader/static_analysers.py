import math
import lizard
import re
from typing import Dict, Any
from typing import List, Dict
from pydriller import Repository

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
        

class LocMiner:
    def __init__(self, repo_path: str):
        """
        repo_path: The local path to the cloned git repository (e.g., './data/raw/mesos_repo')
        """
        self.repo_path = repo_path
        # Regex to catch standard Jira IDs like MESOS-1234, SPARK-999
        # TODO: Account for REGEX EDGE-CASES
        self.jira_regex = re.compile(r'([A-Z]+-\d+)')

    def mine_loc_for_issues(self, target_jira_ids: List[str]) -> Dict[str, int]:
        """
        Scans the git history and maps total LOC (added + removed) to the provided Jira IDs.
        """
        print(f"[*] Starting Git repository mining at: {self.repo_path}")
        
        # Initialize dictionary with 0 LOC for all our target issues
        loc_map = {issue_id: 0 for issue_id in target_jira_ids}
        
        # Traverse every commit in the repository
        for commit in Repository(self.repo_path).traverse_commits():
            
            # Find any Jira IDs mentioned in the commit message
            mentioned_ids = self.jira_regex.findall(commit.msg)
            
            # If the commit mentions an ID we care about, calculate the LOC
            for issue_id in mentioned_ids:
                if issue_id in loc_map:
                    # PyDriller automatically counts these across all files in the commit
                    loc_added = commit.insertions
                    loc_removed = commit.deletions
                    
                    # Your requirement: Positive Additive Manner
                    total_loc_changed = loc_added + loc_removed
                    
                    loc_map[issue_id] += total_loc_changed
                    
        print("[+] Repository mining complete.")
        return loc_map