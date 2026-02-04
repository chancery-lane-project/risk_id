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

👉 The outputs from this model are currently being tested in a **live prototype** (see [Frontend / Provocatype](#frontend--provocatype) section below). This allows us to test with users whether AI-driven clause suggestions increase confidence and adoption compared to manual searching. The prototype now features **live contract analysis** that processes contracts in real-time.  

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
    app.py                   # Main application logic (FastAPI)
    utils.py                 # Utility functions
    complexity.py            # Complexity analysis utilities
    __init__.py              # Package init
    risk_taxonomy.html       # Risk taxonomy reference (HTML)

    models/
      clustering_model.pkl   # Pre-trained clustering model
      umap_model.pkl         # Pre-trained UMAP model
      CC_BERT/               # Climate Contract BERT models (download required - see Installation)
      wandb/                 # Experiment tracking logs and outputs (Weights & Biases)

    provocotype-1/           # Live frontend prototype
      index.htm              # Entry point for contract upload
      index2.htm             # Results page with analysis
      assets/                # CSS, JavaScript, images, and sample contracts
        css/                 # Stylesheets
        js/                  # JavaScript files
        img/                 # Images and loading animations
        sample-contracts/   # Example contract files

    output/                  # Generated highlighted contract outputs (gitignored)
    data/                    # Data files (gitignored - download required - see Installation)
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
- **Live Frontend / Provocatype**  
  - Dynamic HTML prototype with live contract analysis for testing the clause discovery concept with users (see section below).  

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

### Required Data Files and Models

Before running the application, you need to download and set up the required data files and models:

1. **Data Files**: Download the data folder contents from [Google Drive](https://drive.google.com/file/d/1NjdPJlR8lmlyQd6lttNifq6M-k424EiZ/view?usp=sharing) and extract them into the `tclp/data/` directory.

2. **CC_BERT Model**: Download the CC_BERT model from [Google Drive](https://drive.google.com/file/d/1sTpo9iOjhoCZ1qteLqry8jjezWTanSl_/view) and extract it into the `tclp/models/CC_BERT/` directory.

**Note:** These files are not included in the repository due to size limitations and should be downloaded separately.

---

## Usage

### Running the Application

1. **Configure environment:**

```bash
cp .env.template .env
nano .env  # Update OPENROUTER_API_KEY and other variables
```

2. **Start the FastAPI Backend:**

```bash
poetry run uvicorn tclp.app:app --host 0.0.0.0 --port 8000
```

This will:  
- Load the pre-trained models.  
- Start a FastAPI server that can process contract text.  
- Provide API endpoints for contract analysis and clause recommendations.  
- Serve the frontend files from `tclp/provocotype-1/`.

3. **Access the Frontend:**

Once the server is running, open your web browser and navigate to:
```
http://localhost:8000/
```

The frontend will:
- Allow you to upload a contract file (`.txt`, `.pdf`, `.docx`, `.doc`, `.md`)
- Process the contract in real-time using the backend API
- Display analysis results including climate risk classification, recommended clauses, and emissions impact
- Show highlighted contract text with climate-aligned language identified

### Environment Variables

All configuration is managed through environment variables (see `.env.template`):

- `OPENROUTER_API_KEY`: Required for clause recommendations
- `OPENROUTER_MODEL`: LLM model to use (default: `tngtech/deepseek-r1t2-chimera:free`)
- `BASE_PATH`: Subfolder deployment path (e.g., `/risk-id` for serving at `/risk-id/`)
- `TOKENIZERS_PARALLELISM`: Suppress tokenizer warnings (default: `false`)

---

## Frontend / Provocatype

This repository includes a **live frontend prototype** that explores a design hypothesis for helping lawyers and contract drafters identify the most relevant and impactful climate-aligned clauses.

### Hypothesis

> **If users are given a tool that analyses the language of their contract, suggests relevant clauses, and ranks these based on potential emissions reductions, then they will use more impactful climate-aligned clauses compared to searching manually and feel more confident that they have identified the most relevant and effective clause for their context.**

This provocatype tests whether **automated analysis and ranking** can shift behaviour from manual searching to guided clause adoption, increasing both **impact** and **confidence**.

### How It Works

- The prototype features **live contract analysis** that processes uploaded contracts in real-time.  
- Users upload a `.txt` contract file through the web interface.  
- The backend processes the contract using machine learning models to:  
  - Classify the contract's climate-aligned language (unlikely, possible, likely, very likely)  
  - Identify specific climate-aligned clauses in the contract text  
  - Recommend relevant TCLP clauses from the library  
  - Rank clauses by potential emissions reduction impact  
  - Generate highlighted contract text showing where climate language appears  
- Results are displayed with:  
  - A visual gauge showing the climate alignment score  
  - Recommended clauses with explanations  
  - Emissions sources addressed by each clause  
  - Interactive "Show emissions" toggles for detailed information  
  - Links to view and download full clause text  

### Purpose

This is a **provocatype**:  
- Not a finished product, but a working prototype for testing assumptions.  
- Used in user research sessions to provoke reactions, gather feedback, and validate or invalidate the hypothesis.  
- Helps us learn whether users prefer guided, ranked clause suggestions over manual search.  
- Demonstrates live AI-driven contract analysis capabilities.  

### Current Features

- **Live Analysis**: Real-time processing of uploaded contract files  
- **Climate Risk Classification**: Automatic scoring of climate-aligned language  
- **Clause Recommendations**: AI-powered suggestions from TCLP's clause library  
- **Emissions Impact Ranking**: Clauses ranked by potential emissions reduction  
- **Highlighted Output**: Visual highlighting of climate-aligned language in contracts  
- **Interactive UI**: Modern interface with loading states and smooth transitions  

### Limitations

It's important to note the following constraints of this prototype:

- **File Format**: Currently only supports `.txt` files  
- **Concept testing**: While functional, it is designed to test the *concept* and gather user feedback  
- **Single environment**: Primarily tested in **Chrome desktop**. Behaviour may differ elsewhere.  
- **Development prototype**: The code is optimized for rapid iteration and user testing, not production deployment.  

### For Testers

1. Start the backend server (see [Usage](#usage) section above).  
2. Open the frontend at `http://localhost:8000/tclp/provocotype-1/index.htm`.  
3. Upload a `.txt` contract file or use one of the sample contracts.  
4. Wait for the analysis to complete (you'll see a loading screen).  
5. Review the results page which shows:  
   - Climate alignment score with visual gauge  
   - Recommended climate-aligned clauses  
   - Emissions sources addressed by each clause  
   - Highlighted contract text showing climate language  
6. Share your feedback on:  
   - Whether the suggestions feel relevant and useful.  
   - How confident you feel in the recommendations compared to manual searching.  
   - What would make the tool more practical for your work.  

---

## Production Deployment

For production deployment with gunicorn and systemd, see [deployment/README.md](deployment/README.md).

### Quick Start (Production)

```bash
# Configure environment
cp .env.template .env
nano .env  # Update all variables

# Install dependencies
poetry install --no-dev

# Run with gunicorn
poetry run gunicorn -c gunicorn.conf.py tclp.app:app
```

### Systemd Service

See [deployment/README.md](deployment/README.md) for systemd service installation and configuration.

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
