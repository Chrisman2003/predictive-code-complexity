"""
Evaluation engine for calculating rank correlation metrics:
- Kendall's Tau (tau)
- Spearman's Rank Correlation (rho)
- Pairwise Accuracy
- Top-1 Match Accuracy
"""
from typing import List, Tuple
import numpy as np
from scipy.stats import kendalltau, spearmanr
from models.ranking_pipeline.schema import EvaluationMetrics


class ExperimentEvaluator:
    """
    Evaluates aggregated predicted rankings against ground-truth SLOC rankings.
    """

    @staticmethod
    def evaluate(
        problem_id: str,
        ground_truth_ranking: List[str],
        predicted_ranking: List[str]
    ) -> EvaluationMetrics:
        """
        Computes metrics comparing predicted vs. ground-truth paradigm orders.
        Rankings are lists of paradigm strings ordered from LEAST to MOST complex.
        """
        assert set(ground_truth_ranking) == set(predicted_ranking), \
            "Ground truth and predicted rankings must contain the exact same paradigms."

        n = len(ground_truth_ranking)
        gt_map = {paradigm: rank for rank, paradigm in enumerate(ground_truth_ranking)}
        pred_map = {paradigm: rank for rank, paradigm in enumerate(predicted_ranking)}

        # Numerical rank vectors
        gt_ranks = [gt_map[p] for p in ground_truth_ranking]
        pred_ranks = [pred_map[p] for p in ground_truth_ranking]

        # 1. Kendall's Tau (tau)
        tau, _ = kendalltau(gt_ranks, pred_ranks)
        if np.isnan(tau):
            tau = 0.0

        # 2. Spearman's Rank Correlation (rho)
        rho, _ = spearmanr(gt_ranks, pred_ranks)
        if np.isnan(rho):
            rho = 0.0

        # 3. Pairwise Accuracy
        # Percentage of pairwise orderings correctly predicted
        correct_pairs = 0
        total_pairs = 0
        paradigms = list(ground_truth_ranking)

        for i in range(n):
            for j in range(i + 1, n):
                p_a, p_b = paradigms[i], paradigms[j]
                
                # In ground truth, p_a comes before p_b (p_a is less complex)
                gt_a_less_b = gt_map[p_a] < gt_map[p_b]
                pred_a_less_b = pred_map[p_a] < pred_map[p_b]

                if gt_a_less_b == pred_a_less_b:
                    correct_pairs += 1
                total_pairs += 1

        pairwise_acc = correct_pairs / total_pairs if total_pairs > 0 else 0.0

        # 4. Top-1 Match Accuracy (Did model identify the simplest paradigm?)
        top1_match = (ground_truth_ranking[0] == predicted_ranking[0])

        return EvaluationMetrics(
            problem_id=problem_id,
            ground_truth_ranking=ground_truth_ranking,
            predicted_ranking=predicted_ranking,
            pairwise_accuracy=pairwise_acc,
            kendall_tau=float(tau),
            spearman_rho=float(rho),
            top1_match=top1_match
        )