# Risk_ID – Contract Climate Risk Identification

This repository contains code and models developed by **The Chancery Lane Project (TCLP)** to identify and analyse climate-related risks in contracts.  

It supports our work to make contracts more effective tools for driving climate action, by automatically flagging risk areas and linking them to TCLP’s library of climate-aligned clauses.  

---

## Purpose

Legal teams face the challenge of spotting climate-related risks hidden in lengthy and complex contracts. This project explores whether **machine learning models** can help:  

- Analyse contract language.  
- Identify climate-related risks using a **risk taxonomy**.  
- Suggest relevant climate-aligned clauses from TCLP’s library.  
- Provide outputs that can be used for prototyping clause-discovery tools.  

👉 The outputs from this model are currently being tested in a **provocatype prototype**: [Clause Discovery Prototype](https://github.com/alaricking/tclp-provocatype-clause-discovery-2?tab=readme-ov-file). This allows us to test with users whether AI-driven clause suggestions increase confidence and adoption compared to manual searching.  

---

## Repository Structure

```
risk_id-main/
  README.md                  # This file
  pyproject.toml             # Project configuration (Poetry)
  poetry.lock                # Dependency lock file
  .python-version            # Python version specification
  .pre-commit-config.yaml    # Pre-commit hooks
  .dockerignore              # Docker ignore rules
  .gitignore                 # Git ignore rules
  .DS_Store                  # macOS system file (ignore)

  tclp/
    app.py                   # Main application logic
    utils.py                 # Utility functions
    __init__.py              # Package init
    risk_taxonomy.html       # Risk taxonomy reference (HTML)

    models/
      clustering_model.pkl   # Pre-trained clustering model
      umap_model.pkl         # Pre-trained UMAP model
      CC_BERT/               # Placeholder for Climate Contract BERT models
      wandb/                 # Experiment tracking logs and outputs (Weights & Biases)
```

---

## Key Components

- **Models**  
  - `clustering_model.pkl` and `umap_model.pkl`: ML models used for clustering and dimensionality reduction.  
  - `CC_BERT/`: Directory for fine-tuned Climate Contract BERT models.  
- **Risk Taxonomy**  
  - `risk_taxonomy.html`: An overview of the taxonomy of risks that the model is trained to identify.  
- **Experiment Tracking**  
  - `wandb/`: Logs and metadata for training and evaluation runs (using [Weights & Biases](https://wandb.ai/)).  

---

## Installation

This project uses **Poetry** for dependency management.  

```bash
# Clone the repo
git clone https://github.com/chancery-lane-project/risk_id.git
cd risk_id

# Install dependencies
poetry install

# Activate environment
poetry shell
```

Ensure you are using the Python version specified in `.python-version`.  

---

## Usage

To run the model and test contract analysis:

```bash
poetry run python tclp/app.py
```

This will:  
- Load the pre-trained models.  
- Process input text (contract language).  
- Output predicted risk clusters and suggested clauses.  

---

## Development

- Pre-commit hooks are configured in `.pre-commit-config.yaml`. Install them with:  
  ```bash
  pre-commit install
  ```
- Docker can be used for containerised development; see `.dockerignore` for included/excluded files.  

---

## License

See [LICENSE](LICENSE) for details.  
