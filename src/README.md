Source Code Folder for all program files in the Repository

Dataloader: 
1. config.py:           Central path configurations & benchmark registry
2. fetch_repos.py:      Automated Git cloner/updater
3. static_analysers.py: Code parser (SLOC, Cyclomatic, Halstead)
4. build_dataset.py:    Pipeline orchestrator outputting data/processed/
5. dataset.py           PyTorch Dataset loading from data/processed/

Currently the Dataloader pipeline performs the following 3 Tasks:
1. Data-loading: loading Git input sfiles
2. Data-cleaning: TODO
3. Data-labelling: running static analyzers on input files and storing metrics in JSON