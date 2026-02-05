import hashlib
import os
import pickle
import shutil
import time

# Load environment variables first
from dotenv import load_dotenv
load_dotenv()

BASE_PATH = os.getenv("BASE_PATH", "").rstrip("/")

# Suppress tokenizers parallelism warning - read from env
os.environ["TOKENIZERS_PARALLELISM"] = os.getenv("TOKENIZERS_PARALLELISM", "false")

import joblib
import pandas as pd
import torch
import utils
import task_manager
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI

app = FastAPI(
    title="TCLP Risk ID API",
    description="Contract Climate Risk Identification API"
)
CAT0 = "unlikely"
CAT1 = "possible"
CAT2 = "likely"
CAT3 = "very likely"
DEFAULT_MODEL = "tngtech/deepseek-r1t2-chimera:free"

# Enable CORS for frontend-backend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Absolute paths for directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
temp_dir = os.path.join(BASE_DIR, "temp_uploads")
# Make output persist outside temp_dir so cleanup doesn't remove files we link to
output_dir = os.path.join(BASE_DIR, "output")
MODEL_PATH = os.path.join(BASE_DIR, "models", "CC_BERT", "CC_model_detect")
CLAUSE_FOLDER = os.path.join(BASE_DIR, "data", "cleaned_content")
CLAUSE_HTML = os.path.join(BASE_DIR, "data", "clause_boxes")
CLAUSE_TAGS = os.path.join(BASE_DIR, "data", "clause_tags_with_clusters.xlsx")
EMISSION_INDICATORS = os.path.join(BASE_DIR, "data", "full_emissions_table_2.csv")
INDEX_PATH = os.path.join(BASE_DIR, "provocotype-1", "index.htm")
ALT_INDEX_PATH = os.path.join(BASE_DIR, "provocotype-1", "index2.htm")
CLUSTERING_MODEL = os.path.join(BASE_DIR, 'models', 'clustering_model.pkl')
UMAP_MODEL = os.path.join(BASE_DIR, 'models', 'umap_model.pkl')

app.mount(
    "/assets",
    StaticFiles(directory=os.path.join(BASE_DIR, "provocotype-1", "assets")),
    name="assets",
)

os.makedirs(output_dir, exist_ok=True)
app.mount("/output", StaticFiles(directory=output_dir), name="output")

print("[INFO] Loading model and data...")
tokenizer, d_model, c_model, names, docs, final_df, child_names, name_to_child, name_to_url = utils.getting_started(MODEL_PATH, CLAUSE_FOLDER, CLAUSE_HTML)
clause_tags = pd.read_excel(CLAUSE_TAGS)
emission_df = pd.read_csv(EMISSION_INDICATORS)
with open(CLUSTERING_MODEL, 'rb') as f:
    clf = pickle.load(f)
umap_model = joblib.load(UMAP_MODEL)
device = torch.device("cpu")

# Simple in-memory cache for embeddings (keyed by file content hash)
# This allows process_contract and find_clauses to share embeddings
embedding_cache = {}

# Initialize task database
print("[INFO] Initializing task database...")
task_manager.init_db()
print("[INFO] Task database ready")

# --- OpenAI / OpenRouter Setup ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL)  
if not OPENROUTER_API_KEY:
    print("[WARNING] OPENROUTER_API_KEY not found in environment variables. API calls will fail.")
else:
    print(f"[INFO] OpenRouter API key found (starts with: {OPENROUTER_API_KEY[:10]}...)")
    print(f"[INFO] Using OpenRouter model: {OPENROUTER_MODEL}")

client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1")

@app.post("/process/")
async def process_contract(file: UploadFile, background_tasks: BackgroundTasks):
    """
    Endpoint to process a contract file.
    Returns immediately with a task_id for polling.
    """
    # Check validity
    allowed_extensions = ['.txt', '.pdf', '.docx', '.doc', '.md']
    if not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
        return JSONResponse(
            content={
                "error": f"Only {', '.join(allowed_extensions)} files are supported."
            },
            status_code=400,
        )

    # Read file content
    file_content = await file.read()

    # Create task
    task_id = task_manager.create_task()
    print(f"[INFO] Created task {task_id} for file: {file.filename}")

    # Start background processing
    import background_tasks as bg_tasks
    background_tasks.add_task(
        bg_tasks.process_contract_task,
        task_id=task_id,
        file_content=file_content,
        filename=file.filename,
        temp_dir=temp_dir,
        output_dir=output_dir,
        tokenizer=tokenizer,
        d_model=d_model,
        c_model=c_model,
        embedding_cache=embedding_cache,
        CAT0=CAT0,
        CAT1=CAT1,
        CAT2=CAT2,
        CAT3=CAT3
    )

    return {"task_id": task_id, "status": "processing"}

@app.get("/task/{task_id}")
def get_task_status(task_id: str):
    """
    Get the status of a background task.
    Returns task status, progress, and result (if completed).
    """
    task = task_manager.get_task(task_id)

    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    # Return task info
    response = {
        "task_id": task_id,
        "status": task["status"],
        "progress": task.get("progress", 0),
        "created_at": task["created_at"],
        "updated_at": task["updated_at"]
    }

    if task["status"] == "completed":
        response["result"] = task["result"]
    elif task["status"] == "failed":
        response["error"] = task["error"]

    return response


@app.post("/find_clauses/")
async def find_clauses(file: UploadFile, background_tasks: BackgroundTasks):
    """
    Endpoint to find matching clauses for a contract.
    Returns immediately with a task_id for polling.
    """
    # Check validity
    allowed_extensions = ['.txt', '.pdf', '.docx', '.doc', '.md']
    if not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
        return JSONResponse(
            content={
                "error": f"Only {', '.join(allowed_extensions)} files are supported."
            },
            status_code=400,
        )

    # Read file content
    content = await file.read()

    # Create task
    task_id = task_manager.create_task()
    print(f"[INFO] Created find_clauses task {task_id} for file: {file.filename}")

    # Start background processing
    import background_tasks as bg_tasks
    background_tasks.add_task(
        bg_tasks.find_clauses_task,
        task_id=task_id,
        file_content=content,
        filename=file.filename,
        tokenizer=tokenizer,
        c_model=c_model,
        clause_tags=clause_tags,
        clf=clf,
        umap_model=umap_model,
        docs=docs,
        names=names,
        name_to_child=name_to_child,
        name_to_url=name_to_url,
        emission_df=emission_df,
        client=client,
        OPENROUTER_MODEL=OPENROUTER_MODEL,
        DEFAULT_MODEL=DEFAULT_MODEL
    )

    return {"task_id": task_id, "status": "processing"}

@app.get("/", response_class=FileResponse)
def read_root():
    if not os.path.exists(INDEX_PATH):
        raise RuntimeError(f"{INDEX_PATH} not found")
    return FileResponse(INDEX_PATH, media_type="text/html")

# Optional: serve the secondary frontend directly
@app.get("/index2.htm", response_class=FileResponse)
def read_index2_htm():
    if not os.path.exists(ALT_INDEX_PATH):
        raise RuntimeError(f"{ALT_INDEX_PATH} not found")
    return FileResponse(ALT_INDEX_PATH, media_type="text/html")

@app.get("/index2", response_class=FileResponse)
def read_index2():
    if not os.path.exists(ALT_INDEX_PATH):
        raise RuntimeError(f"{ALT_INDEX_PATH} not found")
    return FileResponse(ALT_INDEX_PATH, media_type="text/html")

# --- Run with Uvicorn ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000)