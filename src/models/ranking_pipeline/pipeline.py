"""
Execution logic for the Pairwise Ranking Pipeline.
Called via the global CLI (src/cli.py) or standalone execution.
"""

from typing import List, Dict, Optional, Tuple

from src.models.ranking_pipeline.adapter import HPCDatasetLoader
from src.models.ranking_pipeline.evaluator import ExperimentEvaluator
from src.models.ranking_pipeline.ollama_client import OllamaClient
from src.models.ranking_pipeline.ranking_aggregator import RankingAggregator
from src.models.ranking_pipeline.context_builder import ContextManager
from src.models.ranking_pipeline.schema import PairwisePrediction


# Default benchmark dataset using actual C++ snippets to test SLOC calculation
SAMPLE_HPC_DATASET = [
    {
        "problem_id": "babelstream_dot",
        "title": "BabelStream Memory Bandwidth Dot Product",
        "description": (
            "Compute the dot product of two vectors (A and B) of length N on a GPU device, "
            "and reduce the local sums into a single scalar result on the host."
        ),
        "implementations": {
            "OpenMP_Offload": """
                #pragma omp target teams distribute parallel for reduction(+:sum)
                for (int i = 0; i < N; i++) { sum += A[i] * B[i]; }
            """,
            "CUDA": """
                __global__ void dot_kernel(float *a, float *b, float *c, int n) {
                    __shared__ float cache[256];
                    int tid = threadIdx.x + blockIdx.x * blockDim.x;
                    int cacheIndex = threadIdx.x;
                    float temp = 0;
                    while (tid < n) { temp += a[tid] * b[tid]; tid += blockDim.x * gridDim.x; }
                    cache[cacheIndex] = temp;
                    __syncthreads();
                    int i = blockDim.x/2;
                    while (i != 0) {
                        if (cacheIndex < i) cache[cacheIndex] += cache[cacheIndex + i];
                        __syncthreads();
                        i /= 2;
                    }
                    if (cacheIndex == 0) c[blockIdx.x] = cache[0];
                }
            """,
            "SYCL": """
                q.submit([&](sycl::handler &h) {
                    auto sum_reduction = sycl::reduction(sum_buf, h, sycl::plus<>());
                    h.parallel_for(sycl::range<1>{N}, sum_reduction, [=](sycl::id<1> idx, auto &sum) {
                        sum += a_buf[idx] * b_buf[idx];
                    });
                });
            """,
            "Kokkos": """
                double sum = 0;
                Kokkos::parallel_reduce("DotProduct", N, KOKKOS_LAMBDA(const int i, double& lsum) {
                    lsum += A(i) * B(i);
                }, sum);
            """
        }
    },
    {
        "problem_id": "babelstream_triad",
        "title": "BabelStream Memory Bandwidth Triad",
        "description": (
            "Perform element-wise vector scaling and addition: C[i] = A[i] + scalar * B[i] "
            "for vector size N on device memory."
        ),
        "implementations": {
            "OpenMP_Offload": """
                #pragma omp target teams distribute parallel for
                for (int i = 0; i < N; i++) { C[i] = A[i] + scalar * B[i]; }
            """,
            "CUDA": """
                __global__ void triad_kernel(const float* A, const float* B, float* C, float scalar, int N) {
                    int i = blockIdx.x * blockDim.x + threadIdx.x;
                    if (i < N) { C[i] = A[i] + scalar * B[i]; }
                }
            """,
            "SYCL": """
                q.parallel_for(sycl::range<1>{N}, [=](sycl::id<1> i) {
                    C[i] = A[i] + scalar * B[i];
                }).wait();
            """,
            "Kokkos": """
                Kokkos::parallel_for("Triad", N, KOKKOS_LAMBDA(const int i) {
                    C(i) = A(i) + scalar * B(i);
                });
            """
        }
    }
]


def get_research_context(repo_url: Optional[str] = None) -> ContextManager:
    """
    Constructs and initializes the ContextManager with global, local, and historical knowledge.
    """
    ctx = ContextManager()
    
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
    print("=" * 80)
    print("STARTING GPU PARADIGM COMPLEXITY RANKING PIPELINE")
    print(f"Model: {model_name} | Mode: {'Few-Shot' if use_few_shot else 'Zero-Shot'}")
    if repo_url:
        print(f"Target Repository: {repo_url}")
    print("=" * 80)

    context_manager = get_research_context(repo_url=repo_url)
    loader = HPCDatasetLoader(SAMPLE_HPC_DATASET)
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

    for problem in loader.get_problems():
        print(f"\n[+] Processing Problem: {problem.problem_id}")
        print(f"    Description: {problem.description[:100]}...")

        pairs: List[Tuple[str, str]] = loader.generate_pairwise_combinations(problem.paradigms)
        pairwise_results: List[PairwisePrediction] = []

        for p_a, p_b in pairs:
            print(f"    -> Evaluating pair: ({p_a} vs {p_b})...", end="", flush=True)

            dynamic_context = context_manager.build_context_preamble(p_a, p_b)

            result = ollama_client.predict_pairwise(
                problem_description=problem.description,
                paradigm_a=p_a,
                paradigm_b=p_b,
                context_preamble=dynamic_context,
                few_shot_examples=few_shot_examples
            )

            if isinstance(result, dict):
                result = PairwisePrediction(
                    problem_id=problem.problem_id,
                    paradigm_a=p_a,
                    paradigm_b=p_b,
                    winner=result.get("winner") or result.get("more_complex_paradigm", p_a),
                    reasoning=result.get("reasoning", "Fallback execution due to connection error."),
                    raw_response=str(result)
                )

            pairwise_results.append(result)
            print(f" Winner: {result.winner}")

        aggregated = RankingAggregator.aggregate(
            problem_id=problem.problem_id,
            paradigms=problem.paradigms,
            predictions=pairwise_results
        )
        predicted_ranking: List[str] = aggregated.predicted_ranking

        ground_truth_ranking: List[str] = sorted(
            problem.ground_truth_sloc.keys(),
            key=lambda k: problem.ground_truth_sloc[k],
            reverse=True
        )

        metrics = evaluator.evaluate_problem_ranking(
            problem_id=problem.problem_id,
            predicted_ranking=predicted_ranking,
            ground_truth_ranking=ground_truth_ranking
        )
        all_evaluation_metrics.append(metrics)

        print("\n    [RESULTS SUMMARY]")
        print(f"    Predicted Ranking (Most -> Least Complex): { ' > '.join(predicted_ranking) }")
        print(f"    Ground Truth Ranking (SLOC-based):        { ' > '.join(ground_truth_ranking) }")
        print(
            f"    Kendall's Tau: {metrics['kendall_tau']:.4f} | "
            f"Spearman's Rho: {metrics['spearman_rho']:.4f} | "
            f"Pairwise Acc: {metrics['pairwise_accuracy']:.2%} | "
            f"Top-1 Match: {int(metrics['top1_match'])}"
        )

    overall_summary = evaluator.compute_aggregate_metrics(all_evaluation_metrics)

    print("\n" + "=" * 80)
    print("PIPELINE EXECUTION COMPLETE")
    print(f"Mean Kendall's Tau:    {overall_summary['mean_kendall_tau']:.4f}")
    print(f"Mean Spearman's Rho:   {overall_summary['mean_spearman_rho']:.4f}")
    print(f"Mean Pairwise Acc:     {overall_summary['mean_pairwise_accuracy']:.2%}")
    print(f"Mean Top-1 Match Acc:  {overall_summary['mean_top1_match']:.2%}")
    print("=" * 80)

    return overall_summary