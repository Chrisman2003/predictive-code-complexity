"""
Evaluation engine for calculating rank correlation metrics:
- Kendall's Tau (tau)
- Spearman's Rank Correlation (rho)
- Pairwise Accuracy
- Top-1 Match Accuracy
"""
import math
from typing import Any, Dict, List
from scipy.stats import kendalltau, spearmanr


class ExperimentEvaluator:
    """
    Evaluates alignment between predicted LLM rankings and ground-truth SLOC rankings.
    """

    def evaluate_problem_ranking(
        self,
        problem_id: str,
        predicted_ranking: List[str],
        ground_truth_ranking: List[str]
    ) -> Dict[str, Any]:
        """
        Computes Kendall's Tau, Spearman's Rho, Pairwise Accuracy, and Top-1 Match Accuracy.
        """
        paradigms = sorted(ground_truth_ranking)

        pred_ranks = [predicted_ranking.index(p) for p in paradigms]
        gt_ranks = [ground_truth_ranking.index(p) for p in paradigms]

        tau, _ = kendalltau(pred_ranks, gt_ranks)
        rho, _ = spearmanr(pred_ranks, gt_ranks)

        # Top-1 Accuracy: Did the model correctly identify the #1 item?
        top1_match = 1.0 if predicted_ranking[0] == ground_truth_ranking[0] else 0.0

        # Pairwise Accuracy: Fraction of pairs where relative order matches ground truth
        concordant_pairs = 0
        total_pairs = 0
        n = len(paradigms)

        for i in range(n):
            for j in range(i + 1, n):
                p1, p2 = paradigms[i], paradigms[j]
                pred_order = predicted_ranking.index(p1) < predicted_ranking.index(p2)
                gt_order = ground_truth_ranking.index(p1) < ground_truth_ranking.index(p2)
                
                if pred_order == gt_order:
                    concordant_pairs += 1
                total_pairs += 1

        pairwise_acc = concordant_pairs / total_pairs if total_pairs > 0 else 0.0

        return {
            "problem_id": problem_id,
            "kendall_tau": float(tau) if not math.isnan(tau) else 0.0,
            "spearman_rho": float(rho) if not math.isnan(rho) else 0.0,
            "pairwise_accuracy": pairwise_acc,
            "top1_match": top1_match,
            "predicted_ranking": predicted_ranking,
            "ground_truth_ranking": ground_truth_ranking
        }

    def compute_aggregate_metrics(self, metrics_list: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculates mean metrics across all processed problems.
        """
        total = max(len(metrics_list), 1)
        return {
            "mean_kendall_tau": sum(m["kendall_tau"] for m in metrics_list) / total,
            "mean_spearman_rho": sum(m["spearman_rho"] for m in metrics_list) / total,
            "mean_pairwise_accuracy": sum(m["pairwise_accuracy"] for m in metrics_list) / total,
            "mean_top1_match": sum(m["top1_match"] for m in metrics_list) / total
        }