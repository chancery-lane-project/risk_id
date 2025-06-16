import os

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from openai import OpenAI
from starlette.status import HTTP_401_UNAUTHORIZED

from tclp.clause_recommender import utils

load_dotenv()

# --- FastAPI Setup ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Basic Auth ---
security = HTTPBasic()
USERNAME = "father"
PASSWORD = "christmas"

def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    if not (credentials.username == USERNAME and credentials.password == PASSWORD):
        raise HTTPException(status_code=HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

# --- Load Model + Data ---
MODEL_PATH = "../models/CC_BERT/CC_model_detect"
CLAUSE_FOLDER = "../data/cleaned_content"
CLAUSE_HTML = "../data/clause_boxes"
INDEX_PATH = "frontend/index.html"

print("[INFO] Loading model and data...")
tokenizer, model, names, docs, final_df = utils.getting_started(MODEL_PATH, CLAUSE_FOLDER, CLAUSE_HTML)

# --- OpenAI / OpenRouter Setup ---
client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

# --- Recommender Endpoint ---
@app.post("/recommend")
def recommend_clause(user_input: str, credentials: HTTPBasicCredentials = Depends(verify_credentials)):
    # Step 1: Use BERT model to filter relevant clauses
    top_clauses = utils.filter_clauses(user_input, model, tokenizer, final_df)

    # Step 2: Format for LLM re-ranking
    messages = [
        {"role": "system", "content": "You are a legal assistant helping choose the most relevant climate-aligned clause."},
        {"role": "user", "content": f"Input context: {user_input}\n\nHere are some candidate clauses:\n" + "\n".join(f"{i+1}. {clause}" for i, clause in enumerate(top_clauses)) + "\n\nPlease select the top 3 most relevant clauses."}
    ]

    # Step 3: Call LLM to re-rank
    response = client.chat.completions.create(
        model="deepseek/deepseek-r1-0528-qwen3-8b:free",
        messages=messages,
        temperature=0.1,
        max_tokens=1000
    )

    return {"input": user_input, "llm_response": response.choices[0].message.content}

# --- Health Check ---
@app.get("/ping")
def ping():
    return {"status": "ok"}

# --- Serve Frontend ---
@app.get("/")
def read_root(credentials: HTTPBasicCredentials = Depends(verify_credentials)):
    return FileResponse(INDEX_PATH)

# --- Run with Uvicorn ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
