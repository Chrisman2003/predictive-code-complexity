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
1. config.py:           Central path configurations & benchmark registry
2. fetch_repos.py:      Automated Git cloner/updater
3. static_analysers.py: Code parser (SLOC, Cyclomatic, Halstead)
4. build_dataset.py:    Pipeline orchestrator outputting data/processed/
5. dataset.py           PyTorch Dataset loading from data/processed/

Currently the Dataloader pipeline performs the following 3 Tasks:
1. Data-loading: loading Git input sfiles
2. Data-cleaning: TODO
3. Data-labelling: running static analyzers on input files and storing metrics in JSON


## Model Pipelines: ##
1. Transformer Pipeline for MAP: Problem Description --> Complexity(Implementation Code) 
Training the weights for the model 
--> Collective Repository Training model​ -> Universal Features
--> Target Repository​ Training -> Repository-Specific Features
M_global --> M_local
(Problem Description, Programming Model, Repository) --> Expected Complexity

Research Goal: Analyze Progression in Predictive Power
Linear/Ridge --> Random Forest / Gradient Boosting --> BERT + Regression --> LLM / Transformer + OLLAMA

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

2. OLLAMA Pipeline [Pairwise Comparisons] for MAP: Problem Description --> Ranking(Complexity(Programming Paradigms))
--> HPC: CUDA, Kokkos, OPENCL
--> General: Python, C++, C, Java, etc.
--> Context: Static (Universal Information), Dynamic (Local Information to Repository / Task Description)

3. OLLAMA Pipeline for Partially Generating new Data for MAP: Problem Description --> Complexity(Implementation Code)
Improve Cross-Project predictive capacity