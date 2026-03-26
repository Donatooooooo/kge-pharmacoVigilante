# Knowledge Graph Embeddings for Pharmacovigilance

Drug safety post-marketing surveillance relies on spontaneous reporting systems that suffer from severe under-reporting and slow signal accumulation. As a result, adverse events, particularly rare ones, may remain undetected for years after a drug reaches the market. This work investigates whether the relational structure of pharmaceutical knowledge graphs can support early side effect prediction. We formalize the problem as a link prediction task and compare three approaches: a marginal frequency baseline, direct link prediction using knowledge graph embeddings with side effects included in the graph, and an indirect method where embeddings are learned from a side-effect-free graph and used as features for a downstream ranking model. Experiments show that the indirect approach outperforms both alternatives, achieving a MRR of 0.318 and Hits@10 of 84.8%. Further stratified analysis shows that this advantage is most pronounced for rare side effects, precisely the regime where spontaneous reporting systems are most deficient. These results suggest that knowledge graph embeddings capture latent pharmacological signals, enabling prioritized screening of candidate adverse events for drugs with incomplete safety profiles.

## Project Organization

```
├── README.md
├── pyproject.toml
├── uv.lock
│
├── data                                    <- Data directory, DVC-tracked.
│   ├── raw                                 <- The original, immutable data dump.
│   ├── processed                           <- The final, canonical data sets for modeling.
│   └── graph                               <- Graph-derived artifacts.
│       └── embeddings.pkl
│
├── models                                  <- Trained and serialized models, DVC-tracked.
│   ├── learned_kge                         <- PyKEEN KGE model.
│   ├── learned_kge_se                      <- PyKEEN KGE side effect model.
│   ├── umls                                <- UMLS concept cache.
│   └── xgboost                             <- XGBoost model.
│
├── notebooks                               <- Jupyter EDA notebook.
├── reports                                 <- Generated analysis outputs.
│
└── src                                     <- Source code for use in this project.
    ├── __init__.py
    ├── config.py                           <- Project-wide configuration and paths.
    ├── run.py                              <- Main entry point for the pipeline.
    ├── data                                <- Scripts for data processing.
    │   ├── labels.py                       <- Label cleaning pipeline.
    │   ├── make_dataset.py                 <- Data preprocessing and merge.
    │   ├── patterns.py                     <- Usefull patterns.
    │   └── __init__.py
    ├── graph                               <- Knowledge graph.
    │   ├── build_graph.py                  <- Graph builder.
    │   ├── embeddings.py                   <- Downstream dataset.
    │   ├── hpo.py                          <- KGE hpo pipeline.
    │   └── __init__.py
    └── modeling                            <- Model training and evaluation.
        ├── baseline.py                     <- Marginal frequency baseline.
        ├── kge.py                          <- Approach A: direct link prediction.
        ├── xgboost_cv.py                   <- Approach B: downstream ranking.
        ├── __init__.py
        └── util
            ├── data_loader.py              <- Dataset loader.
            ├── evaluation.py               <- Evaluation methods.
            ├── kge_scorer.py               <- KGE side effect score.
            └── __init__.py
