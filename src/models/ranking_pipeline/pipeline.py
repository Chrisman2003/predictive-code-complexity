"""
Execution logic for the Pairwise Ranking Pipeline.
Called via the global CLI (src/cli.py) or standalone execution.
"""

import json
from typing import List, Dict, Optional, Tuple

from models.ranking_pipeline.data_loader import HPCDatasetLoader
from models.ranking_pipeline.evaluator import ExperimentEvaluator
from models.ranking_pipeline.ollama_client import OllamaClient
from models.ranking_pipeline.ranking_aggregator import RankingAggregator
from models.ranking_pipeline.context_builder import ContextManager
from models.ranking_pipeline.schema import PairwiseComparisonResult


# Default benchmark dataset for GPU paradigm complexity testing
SAMPLE_HPC_DATASET = [
    {
        "problem_id": "babelstream_dot",
        "description": (
            "Compute the dot product of two vectors (A and B) of length N on a GPU device, "
            "and reduce the local sums into a single scalar result on the host."
        ),
        "paradigms": ["CUDA", "SYCL", "OpenMP_Offload", "Kokkos"],
        "ground_truth_sloc": {
            "OpenMP_Offload": 15,
            "CUDA": 32,
            "SYCL": 48,
            "Kokkos": 55
        }
    },
    {
        "problem_id": "babelstream_triad",
        "description": (
            "Perform element-wise vector scaling and addition: C[i] = A[i] + scalar * B[i] "
            "for vector size N on device memory."
        ),
        "paradigms": ["CUDA", "SYCL", "OpenMP_Offload", "Kokkos"],
        "ground_truth_sloc": {
            "OpenMP_Offload": 10,
            "CUDA": 25,
            "SYCL": 38,
            "Kokkos": 42
        }
    }
]


def get_research_context(repo_url: Optional[str] = None) -> ContextManager:
    """
    Constructs and initializes the ContextManager with global, local, and historical knowledge.
    """
    ctx = ContextManager(docs_dir="./data/docs")
    
    global_data = {
        "descriptions": {
            "CUDA": "Proprietary, low-level extension of C++ by NVIDIA. Highly explicit kernel launches.",
            "SYCL": "High-level, single-source C++ standard relying heavily on templates, accessors, and queues.",
            "OpenMP_Offload": "Directive-based (pragma) extension to C/C++. Avoids explicit kernel launcher boilerplate.",
            "Kokkos": "C++ performance portability framework. Encapsulates execution spaces and multi-dimensional Views."
        },
        "memory_models": {
            "CUDA": "Requires explicit cudaMalloc and cudaMemcpy, or cudaMallocManaged.",
            "SYCL": "Uses abstraction via sycl::buffer and sycl::accessor, or explicit Unified Shared Memory (USM).",
            "OpenMP_Offload": "Handled implicitly or explicitly via `#pragma omp target data map(...)` directives.",
            "Kokkos": "Handled via Kokkos::View encapsulating layout and memory spaces."
        }
    }

    local_data = {
        "repository_name": repo_url if repo_url else "BabelStream",
        "repository_goals": "Measure fundamental memory bandwidth across HPC programming models using minimal operational kernels.",
        "architectural_constraints": "Implementations must follow standard C++17 and fit within single translation units where possible."
    }

    historical_data = {
        "maturity": {
            "CUDA": "Ubiquitous, gold standard GPU ecosystem with mature toolchain support.",
            "SYCL": "Standardized and modern, though compiler implementations (DPC++, AdaptiveCpp) vary in error verbosity.",
            "OpenMP_Offload": "Standardized across GCC/LLVM, but target offloading edge cases can yield difficult compiler errors.",
            "Kokkos": "Extremely mature in US National Labs; requires CMake toolchain integration overhead."
        },
        "known_frictions": {
            "CUDA": "Manual thread block/grid index calculations increase structural SLOC.",
            "SYCL": "Buffer/accessor patterns add high boilerplate overhead for basic memory transfers.",
            "OpenMP_Offload": "Opaque pragmas simplify syntax but complicate debugging when mapping fails.",
            "Kokkos": "Heavy template metaprogramming yields verbose error backtraces and learning curve."
        }
    }

    ctx.load_from_dictionaries(global_data, local_data, historical_data)
    return ctx


def run_ranking_pipeline(
    model_name: str = "qwen2.5-coder", 
    use_few_shot: bool = False, 
    repo_url: Optional[str] = None
) -> Dict[str, float]:
    """
    Executes the full pairwise ranking experiment:
    1. Prepares Context and Dataset.
    2. Runs pairwise comparisons via Ollama.
    3. Aggregates pairwise probabilities into continuous complexity scores.
    4. Evaluates predicted rankings against ground truth SLOC using Kendall Tau and Spearman Correlation.
    """
    print("=" * 80)
    print("STARTING GPU PARADIGM COMPLEXITY RANKING PIPELINE")
    print(f"Model: {model_name} | Mode: {'Few-Shot' if use_few_shot else 'Zero-Shot'}")
    if repo_url:
        print(f"Target Repository: {repo_url}")
    print("=" * 80)

    # 1. Initialize Context Manager and Dataset Loader
    context_manager = get_research_context(repo_url=repo_url)
    loader = HPCDatasetLoader(dataset_raw=SAMPLE_HPC_DATASET)
    ollama_client = OllamaClient(model_name=model_name)
    evaluator = ExperimentEvaluator()

    few_shot_examples = None
    if use_few_shot:
        few_shot_examples = [
            {
                "problem": "Simple vector addition on GPU.",
                "paradigm_a": "CUDA",
                "paradigm_b": "OpenMP_Offload",
                "winner": "CUDA",
                "reasoning": "CUDA requires explicit kernel definition, grid configuration, and host-device memory allocations, making it inherently more complex than a single OpenMP pragma."
            }
        ]

    all_evaluation_metrics = []

    # 2. Iterate through problems in the dataset
    for problem in loader.get_problems():
        print(f"\n[+] Processing Problem: {problem.problem_id}")
        print(f"    Description: {problem.description[:100]}...")
        
        pairs: List[Tuple[str, str]] = loader.generate_pairwise_combinations(problem.paradigms)
        pairwise_results: List[PairwiseComparisonResult] = []

        # 3. Perform pairwise inference
        for p_a, p_b in pairs:
            print(f"    -> Evaluating pair: ({p_a} vs {p_b})...", end="", flush=True)
            
            # Build dynamic knowledge context for the pair
            dynamic_context = context_manager.build_context_preamble(p_a, p_b)
            
            # Predict complexity relationship
            result = ollama_client.predict_pairwise(
                problem_description=problem.description,
                paradigm_a=p_a,
                paradigm_b=p_b,
                context_preamble=dynamic_context,
                few_shot_examples=few_shot_examples
            )
            
            pairwise_results.append(result)
            print(f" Winner: {result.more_complex_paradigm} (Confidence: {result.confidence_score})")

        # 4. Aggregate pairwise comparisons into a global ranking
        aggregator = RankingAggregator(paradigms=problem.paradigms, comparisons=pairwise_results)
        predicted_scores: Dict[str, float] = aggregator.compute_bradley_terry_scores()
        predicted_ranking: List[str] = aggregator.get_sorted_ranking(predicted_scores)

        # 5. Determine ground truth ranking from SLOC metrics
        ground_truth_ranking: List[str] = sorted(
            problem.ground_truth_sloc.keys(),
            key=lambda k: problem.ground_truth_sloc[k],
            reverse=True  # Higher SLOC = Higher Complexity
        )

        # 6. Evaluate alignment
        metrics = evaluator.evaluate_problem_ranking(
            problem_id=problem.problem_id,
            predicted_ranking=predicted_ranking,
            ground_truth_ranking=ground_truth_ranking,
            predicted_scores=predicted_scores,
            ground_truth_sloc=problem.ground_truth_sloc
        )
        all_evaluation_metrics.append(metrics)

        print("\n    [RESULTS SUMMARY]")
        print(f"    Predicted Ranking (Most -> Least Complex): { ' > '.join(predicted_ranking) }")
        print(f"    Ground Truth Ranking (SLOC-based):        { ' > '.join(ground_truth_ranking) }")
        print(f"    Kendall's Tau: {metrics['kendall_tau']:.4f} | Spearman's Rho: {metrics['spearman_rho']:.4f}")

    # 7. Overall Pipeline Summary
    overall_summary = evaluator.compute_aggregate_metrics(all_evaluation_metrics)
    
    print("\n" + "=" * 80)
    print("PIPELINE EXECUTION COMPLETE")
    print(f"Mean Kendall's Tau: {overall_summary['mean_kendall_tau']:.4f}")
    print(f"Mean Spearman's Rho: {overall_summary['mean_spearman_rho']:.4f}")
    print("=" * 80)

    return overall_summary