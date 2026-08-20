Source Code Folder for all program files in the Repository
# Research Problems #
Problem 1: Problem Description --> Ranking on GPU Paradigms for Complexity(Implementation)
Problem 2: Problem Description --> Complexity(Implementation Code)

Examples for Problem Descriptions:
- User Stories: [Agile: Initiative -> Epic -> Story]
- Issue Descriptions

Examples for Complexity Metrics (Interpretable):
- Story Points = f(effort, complexity, uncertainty, risk) <-- Subjective (Human-Labelled)
- Lines of Code <-- Objective (Algorithm-Labelled) 
- Token Count <-- Objective (Algorithm-Labelled)


## Dataloader: ##
Responsibility: Executes Repository level ETL & Data Ingestion
--> It serves as a Data Producer for the rest of all Consumer Model Pipelines

Structure:
1. config.py:           Central path configurations & benchmark registry
2. fetch_repos.py:      Automated Git cloner/updater
3. static_analysers.py: Code parser (SLOC, Cyclomatic, Halstead)
4. build_dataset.py:    Pipeline orchestrator outputting data/processed/

Currently the Dataloader pipeline performs the following 3 Tasks:
1. Data-loading: loading Git input sfiles
2. Data-cleaning: TODO
3. Data-labelling: running static analyzers on input files and storing metrics in JSON


## Model Pipelines: ##
1. Prediction Pipeline 
MAP: Problem Description --> Complexity(Implementation Code) 
Goal: Analyze Progression in Predictive Power 
Linear/Ridge --> Random Forest / Gradient Boosting --> BERT + Regression --> LLM / Transformer + OLLAMA

Training the weights for the model 
--> Collective Repository Training model​ -> Universal Features [Train Pretrained-Encoder with new Head]
--> Target Repository​ Training -> Repository-Specific Features [Finetune Full-Encoder]
M_global --> M_local
(Problem Description, Programming Model, Repository) --> Expected Complexity

Stage 2.2 Focus: The model learns to read CODE to predict complexity
X = item["source_code"] 
Y = item["metrics"]["halstead_effort"]

State 2.3 Focus: The model learns to read the TASK to predict complexity
X = f"Task: {item['problem_description']} | Paradigm: {item['paradigm']}"
Y = item["metrics"]["halstead_effort"]

Problem 2:
Problem Description:
"Implement a memory-bandwidth-bound vector operation on a GPU."

Model output [LOC]:
    1. Kokkos      150
    2. CUDA        118
    3. OpenMP      94
    4. SYCL        72


2. Ranking Pipeline 
PROBLEMS [FIX]: 
- FEW-SHOT
- REPOSITORY URL PARSING
MAP: Problem Description --> Ranking(Complexity(Programming Paradigms))
Goal: for an arbitrary problem description (with global and local context) produce a ranking on the Implementation Complexity across GPU Paradigms

Utilize Pairwise Comparisons with OLLAMA
--> HPC: CUDA, Kokkos, OPENCL
--> General: Python, C++, C, Java, etc.
--> Context: Static (Universal Information), Dynamic (Local Information to Repository / Task Description)
Motivation for Prompting Pairwise Comparisons: 
https://aclanthology.org/2024.findings-naacl.97.pdf
https://dl.acm.org/doi/pdf/10.1145/3626772.3657813

[Processed Data / Raw Benchmark]
               │
               ▼
┌──────────────────────────────────────────────────────────────────┐
│                   pipeline.py (Orchestrator)                     │
└──────┬─────────────┬─────────────┬─────────────┬───────────┬─────┘
       │             │             │             │           │
       ▼             ▼             ▼             ▼           ▼
┌────────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐
│data_loader │ │ context_  │ │ prompts + │ │ ranking_  │ │evaluator  │
│  .py       │ │ builder.py│ │ ollama_   │ │aggregator │ │  .py      │
│            │ │           │ │ client.py │ │  .py      │ │           │
└─────┬──────┘ └─────┬─────┘ └─────┬─────┘ └─────┬─────┘ └─────┬─────┘
      │              │             │             │             │
      ▼              ▼             ▼             ▼             ▼
  Parses entries  Assembles    Executes      Aggregates    Computes 
  and yields      Global/Local pairwise      pairs into    Kendall Tau
  (P_A, P_B)      + RAG        LLM           Bradley-Terry & Spearman
  pairs           preamble     inference     scores        vs SLOC



3. Generative Pipeline 
MAP: Problem Description --> Complexity(Implementation Code)
Goal: Generating new Training Data / Context with OLLAMA Pipelines
Improve Cross-Project predictive capacity