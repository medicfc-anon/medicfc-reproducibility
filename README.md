# MedicFC

This repository contains the resources and implementations used in MedicFC for biomedical claim verification.

The repository provides:

- The HealthFC-based claim dataset (`Datensatz.csv`)
- The final evidence collection and reranking results (`MedicFC.csv`)
- Evidence retrieval and concept-aware reranking scripts
- The multi-encoder verification model.
- The input file to the multi-encoder model (`Input.pkl`)

## Hyperparameter Configuration

The optimal hyperparameters used for training the models are summarized below.

| Hyperparameter | PubMedBERT | DeBERTa-v3-Large | SciBERT |
|---|---:|---:|---:|
| Learning rate | $5\times10^{-5}$ | $5\times10^{-6}$ | $5\times10^{-5}$ |
| Training epochs | 5 | 5 | 5 |
| Weight decay | 0.02 | 0.01 | 0.01 |
| Dropout | 0.25 | Default | 0.10 |
| Max sequence length | 384 | 512 | 512 |