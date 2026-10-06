# MedicFC

This repository contains the resources and implementations used in MedicFC for biomedical claim verification.

The repository provides:

- The HealthFC-based claim dataset (`Datensatz.csv`)
- The final evidence collection and reranking results (`MedicFC.csv`)
- Evidence retrieval and concept-aware reranking scripts
- The multi-encoder verification model.
- The input file to the multi-encoder model (`Input.pkl`)

## Repository Structure

```text
medicfc-evidence-retrieval/

├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── config/
│   └── config.example.yaml
├── data/
│   ├── Datensatz.csv
│   └── README.md
├── outputs/
│   ├── MedicFC.csv
│   └── README.md
├── scripts/
│   └── run_pipeline.py
└── src/
    ├── preprocessing.py
    ├── phrase_extraction.py
    ├── umls.py
    ├── query_builder.py
    ├── google_search.py
    ├── web_extraction.py
    ├── sentence_ranking.py
    ├── reranking.py
    └── four_encoder/
        ├── model.py
        ├── train.py
        └── evaluate.py