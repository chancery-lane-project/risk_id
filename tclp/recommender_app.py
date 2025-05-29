import os
import pickle
import re
from difflib import get_close_matches
from pathlib import Path

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from starlette.status import HTTP_401_UNAUTHORIZED  # Import the status code

from tclp.clause_recommender import utils

app = FastAPI()

# Enable CORS for your frontend to access backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Basic authentication
security = HTTPBasic()

# Dummy credentials for demonstration purposes
USERNAME = "father"
PASSWORD = "christmas"

# Verify credentials
def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = credentials.username == USERNAME
    correct_password = credentials.password == PASSWORD
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )

# Load model and embeddings
REPO_ROOT = Path(__file__).resolve().parents[2]  # go up from tclp/clause_recommender/

THIS_DIR = Path(__file__).resolve().parent
INDEX_PATH = THIS_DIR / "index.html"

DATA_DIR = os.environ.get("TCLP_DATA_DIR", REPO_ROOT / "tclp/data")
MODEL_DIR = os.environ.get("TCLP_MODEL_DIR", REPO_ROOT / "tclp/CC_BERT/CC_model")
CLUSTER_PATH = os.environ.get("TCLP_CLUSTER_MODEL", REPO_ROOT / "tclp/clause_recommender/clustering_model.pkl")
CLAUSES_HTML = os.environ.get("TCLP_CLAUSE_BOXES", REPO_ROOT / "tclp/data/clause_boxes")
CLAUSES_DIR = os.environ.get("TCLP_CLAUSES_DIR", REPO_ROOT / "tclp/data/cleaned_content")

tokenizer, model, names, docs, final_df = utils.getting_started(MODEL_DIR, CLAUSES_DIR, CLAUSES_HTML)

clause_tags = pd.read_excel(os.path.join(DATA_DIR, "clause_tags_with_clusters.xlsx"))

documents, file_names, clause_names = utils.load_clauses(CLAUSES_HTML)

texts = clause_tags['CombinedText'].tolist()

with open(CLUSTER_PATH, 'rb') as f:
    clustering_model = pickle.load(f)


@app.post("/find_clauses/")
async def find_clauses(file: UploadFile):
    content = await file.read()
    method = "mean"
    query = content.decode("utf-8")
    query_embedding = utils.encode_text(query, tokenizer, model, method).reshape(1, -1)
    
    subset_docs, subset_names, cluster_subset_df = utils.perform_cluster(clustering_model, query_embedding, tokenizer, model, clause_tags)
    # could be good to load the embeddings in rather than having to do this every time
    top_names_sem, top_scores_sem, top_texts = utils.get_embedding_matches_subset(
        query_text=query,
        documents_subset=subset_docs,
        names_subset=subset_names,
        tokenizer=tokenizer,
        model=model,
        method='cls',
        k=10
    )

    best_match_names = top_names_sem[:3]
    print("Best match names:", best_match_names)
    best_match_scores = top_scores_sem[:3]
    
    
    return {
        "matches": [
            {"name": name.replace(".txt", ""), "score": float(score)}
            for name, score in zip(best_match_names, best_match_scores)
        ]
    }

# allow the user to see the clause

def normalize(name):
    return re.sub(r"[^\w]+", "_", name.lower()).strip("_")

@app.get("/clause/{clause_name}")
def get_clause(clause_name: str):
    try:
        # Fuzzy match against full HTML titles
        close_matches = get_close_matches(clause_name, clause_names, n=1, cutoff=0.8)
        if not close_matches:
            raise HTTPException(status_code=404, detail="Clause not found")

        best_title = close_matches[0]
        index = clause_names.index(best_title)

        return {
            "name": best_title,
            "text": documents[index]
        }

    except Exception as e:
        print(f"Clause lookup error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/")
def read_root(credentials: HTTPBasicCredentials = Depends(verify_credentials)):
    return FileResponse(INDEX_PATH)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)