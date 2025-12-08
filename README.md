# Risk_ID – Contract Climate Risk Identification

This repository contains code and models developed by **The Chancery Lane Project (TCLP)** to identify and analyse climate-related risks in contracts.  

It supports our work to make contracts more effective tools for driving climate action, by automatically flagging risk areas and linking them to TCLP's library of climate-aligned clauses.  

---

## Purpose

Legal teams face the challenge of spotting climate-related risks hidden in lengthy and complex contracts. This project explores whether **machine learning models** can help:  

- Analyse contract language.  
- Identify climate-related risks using a **risk taxonomy**.  
- Suggest relevant climate-aligned clauses from TCLP's library.  
- Provide outputs that can be used for prototyping clause-discovery tools.  

👉 The outputs from this model are currently being tested in a **provocatype prototype** (see [Static Frontend / Provocatype](#static-frontend--provocatype) section below). This allows us to test with users whether AI-driven clause suggestions increase confidence and adoption compared to manual searching.  

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

  # Static Frontend / Provocatype
  index.html                 # Entry point for the static prototype
  contract-report-*.htm      # Example contract analysis reports
  assets/                    # CSS, JavaScript, and images for the static frontend

  tclp/
    app.py                   # Main application logic (FastAPI)
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
- **Static Frontend / Provocatype**  
  - Static HTML prototype for testing the clause discovery concept with users (see section below).  

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

### Running the FastAPI Backend

To run the FastAPI application:

```bash
poetry run uvicorn tclp.app:app --host 0.0.0.0 --port 8000
```

This will:  
- Load the pre-trained models.  
- Start a FastAPI server that can process contract text.  
- Provide API endpoints for contract analysis and clause recommendations.  

### Using the Static Frontend

The static frontend can be used independently for user testing. Simply open `index.html` in a web browser. See the [Static Frontend / Provocatype](#static-frontend--provocatype) section for more details.

---

## Static Frontend / Provocatype

This repository includes a **static frontend prototype** that explores a design hypothesis for helping lawyers and contract drafters identify the most relevant and impactful climate-aligned clauses.

### Hypothesis

> **If users are given a tool that analyses the language of their contract, suggests relevant clauses, and ranks these based on potential emissions reductions, then they will use more impactful climate-aligned clauses compared to searching manually and feel more confident that they have identified the most relevant and effective clause for their context.**

This provocatype tests whether **automated analysis and ranking** can shift behaviour from manual searching to guided clause adoption, increasing both **impact** and **confidence**.

### How It Works

- This prototype is a **simulation of an AI model** that analyses contracts and recommends relevant clauses.  
- The underlying model and approach are documented in this repository (the `risk_id` backend).  
- The text outputs in these static HTML pages are **real results** generated by that model.  
- **index.html** acts as the entry point for the simulation.  
- **contract-report-[n].htm** represent different example outputs from the model:  
  - Each shows suggested TCLP clauses linked to the analysed contract text.  
  - Clauses are ranked by their **potential emissions reduction impact**.  
  - The reasoning is displayed to build user trust and confidence in the suggestions.  

### Purpose

This is a **provocatype**:  
- Not a finished product, but a stimulus for testing assumptions.  
- Used in user research sessions to provoke reactions, gather feedback, and validate or invalidate the hypothesis.  
- Helps us learn whether users prefer guided, ranked clause suggestions over manual search.  

### Limitations

It's important to note the following constraints of this prototype:

- **Static only**: This is a static prototype. No live analysis or contract processing takes place.  
- **Concept testing**: It is designed purely to test the *concept* — i.e. does the idea make sense and provide value to users — not to demonstrate working technology.  
- **Single environment**: It has only been tested in one environment: **Chrome desktop**. Behaviour may differ elsewhere.  
- **Quick and messy code**: The code has been hacked together quickly and should be considered throwaway. It is not production-quality.  

### For Testers

1. Open **index.html** in your web browser (double-click should work on most systems).  
2. Explore the interface and follow the prompts.  
3. View the different **contract report** pages (`contract-report-1.htm` to `contract-report-4.htm`). Each shows:  
   - Suggested climate-aligned clauses.  
   - A ranking of clauses by potential emissions reduction.  
   - Explanations of why each clause is relevant.  
4. Share your feedback on:  
   - Whether the suggestions feel relevant and useful.  
   - How confident you feel in the recommendations compared to manual searching.  
   - What would make the tool more practical for your work.  

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
