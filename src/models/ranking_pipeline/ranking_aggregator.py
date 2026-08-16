"""
Aggregation engine for building a total ranking from pairwise predictions.
Handles cyclic/inconsistent preferences using Copeland's Method with Keener tie-breaking.
"""
from typing import Dict, List, Tuple
import numpy as np
from models.ranking_pipeline.schema import AggregatedRanking, PairwisePrediction


class RankingAggregator:
    """
    Aggregates pairwise tournament outcomes into a complete, consistent total ordering.
    """

    @staticmethod
    def aggregate(
        problem_id: str,
        paradigms: List[str],
        predictions: List[PairwisePrediction]
    ) -> AggregatedRanking:
        """
        Aggregates pairwise comparisons into a single ranking ordered from
        LEAST COMPLEX (Rank 1) to MOST COMPLEX.
        """
        paradigms = sorted(paradigms)
        n = len(paradigms)
        p_to_idx = {p: i for i, p in enumerate(paradigms)}

        # Initialize Pairwise Win Matrix
        # W[i][j] = 1 if paradigm i was predicted LESS complex than paradigm j
        W = np.zeros((n, n), dtype=int)

        win_matrix_dict = {p1: {p2: 0 for p2 in paradigms} for p1 in paradigms}

        for pred in predictions:
            winner = pred.winner
            # The 'winner' is the LESS complex paradigm
            p_a, p_b = pred.paradigm_a, pred.paradigm_b
            loser = p_b if winner == p_a else p_a

            i, j = p_to_idx[winner], p_to_idx[loser]
            W[i][j] = 1
            win_matrix_dict[winner][loser] = 1

        # Check for cyclic preferences (e.g., A > B, B > C, C > A)
        has_cycles = RankingAggregator._detect_cycles(W)

        # Compute Copeland's Score: Wins - Losses
        # Higher Copeland score = Beat more paradigms = Less complex
        wins = np.sum(W, axis=1)
        losses = np.sum(W, axis=0)
        copeland_scores = wins - losses

        copeland_dict = {paradigms[i]: float(copeland_scores[i]) for i in range(n)}

        # Secondary Tie-Breaker: Keener / Win-Ratio Matrix Strength
        # To resolve ties deterministically if Copeland scores are identical due to cycles
        keener_scores = RankingAggregator._compute_keener_scores(W)

        # Sort paradigms: primary key = Copeland Score (descending), secondary key = Keener Score (descending)
        ranked_indices = sorted(
            range(n),
            key=lambda i: (copeland_scores[i], keener_scores[i], -i),
            reverse=True
        )

        predicted_ranking = [paradigms[idx] for idx in ranked_indices]

        return AggregatedRanking(
            problem_id=problem_id,
            predicted_ranking=predicted_ranking,
            copeland_scores=copeland_dict,
            win_matrix=win_matrix_dict,
            has_cycles=has_cycles
        )

    @staticmethod
    def _detect_cycles(W: np.ndarray) -> bool:
        """
        Detects if the tournament graph contains directed cycles using Warshall's algorithm.
        """
        n = W.shape[0]
        reach = W.copy()
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    reach[i][j] = reach[i][j] or (reach[i][k] and reach[k][j])

        # A cycle exists if any node can reach itself
        for i in range(n):
            if reach[i][i]:
                return True
        return False

    @staticmethod
    def _compute_keener_scores(W: np.ndarray) -> np.ndarray:
        """
        Computes Keener strength scores to break ties in cyclic tournaments.
        Uses Laplace smoothing on win-loss ratios.
        """
        n = W.shape[0]
        scores = np.zeros(n)
        for i in range(n):
            total_matches = np.sum(W[i, :]) + np.sum(W[:, i])
            if total_matches > 0:
                # Laplace smoothed ratio
                scores[i] = (np.sum(W[i, :]) + 1.0) / (total_matches + 2.0)
            else:
                scores[i] = 0.5
        return scores