import os
import pickle
import re
from urllib.parse import unquote

import joblib
import pandas as pd
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from openai import OpenAI
from starlette.status import HTTP_401_UNAUTHORIZED

import utils

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
MODEL_PATH = "models/CC_BERT/CC_model_detect"
CLAUSE_FOLDER = "data/cleaned_content"
CLAUSE_HTML = "data/clause_boxes"
CLAUSE_TAGS = "data/clause_tags_with_clusters.xlsx"
INDEX_PATH = "recommender_index.html"
CLUSTERING_MODEL = 'models/clustering_model.pkl'
UMAP_MODEL = 'models/umap_model.pkl'

print("[INFO] Loading model and data...")
tokenizer, model, names, docs, final_df = utils.getting_started(MODEL_PATH, CLAUSE_FOLDER, CLAUSE_HTML)
clause_tags = pd.read_excel(CLAUSE_TAGS)
with open(CLUSTERING_MODEL, 'rb') as f:
    clf = pickle.load(f)
umap_model = joblib.load(UMAP_MODEL)

# --- OpenAI / OpenRouter Setup ---
client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1")


# --- Recommender Endpoint for Frontend ---
@app.post("/find_clauses/")
async def find_clauses(file: UploadFile = File(...)):
    if not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files are supported.")

    try:
        content = await file.read()
        query_text = content.decode('utf-8')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read file: {str(e)}")

    if not query_text.strip():
        raise HTTPException(status_code=400, detail="Uploaded .txt file is empty.")
    
    subset_docs, subset_names, cluster_subset_df = utils.perform_cluster(clf, query_text, tokenizer, model, clause_tags, umap_model, embed = True)
    bow_results = utils.find_top_similar_bow(
    target_doc=query_text,
    documents=docs,
    file_names=names,
    similarity_threshold=0.1, k =20
    )
    top_docs = bow_results["Documents"]
    top_names = bow_results["Top_Matches"]
    top_names_bow, top_scores_bow, top_texts_bow = utils.get_embedding_matches_subset(
        query_text,
        top_docs,
        top_names,
        tokenizer,
        model, k=5
    )
    
    ## putting them into a shared dataframe 
    df_cluster = pd.DataFrame({
        "text": subset_docs,
        "source_name": subset_names,
        "matched_by": ["cluster"] * len(subset_names)
    })

    df_bow = pd.DataFrame({
        "text": top_texts_bow,
        "source_name": top_names_bow,
        "matched_by": ["bow"] * len(top_texts_bow)
    }).head(5)  # Top 5 from BOW

    # Step 2: Combine (without deduplicating)
    combined_df = pd.concat([df_cluster, df_bow], ignore_index=True)
    query_text_short = query_text[:1000]
    
    messages = [
    {
        "role": "system",
        "content": "You are a legal AI assistant that helps review and select climate-aligned clauses for the uploaded document. You can only select from those clauses provided to you. We are trying to help the writers of the document integrate climate-aligned language."
    },
    {
        "role": "user",
        "content": f"Here's the contract:\n\n{query_text_short.strip()}\n\nI will send you some clauses next. For now, just confirm you have read the contract and are ready to receive the clauses. A short summary of the content of the contract would be fine."
    }
    ]
        
    response = client.chat.completions.create(
        model="deepseek/deepseek-r1-0528-qwen3-8b:free", 
        messages= messages,
        temperature=0.1,
        max_tokens=1000
    )

    assistant_reply_1 = response.choices[0].message.content
    messages.append({"role": "assistant", "content": assistant_reply_1})

    clause_block = "Here are the clauses:\n\n"

    for i, row in combined_df.iterrows():
        clause_block += (
            f"Clause {i+1}\n"
            f"Name: {row['source_name']}\n"
            f"Method: {row['matched_by']}\n"
            f"Full Text:\n{row['text']}\n\n"
        )

    clause_block += '''Select the 3 clauses from the list that best align with the contract. Follow these rules:

    1. Your response must be a JSON of exactly three objects, each with the keys "Clause Name" and "Reasoning".
    3. Only select from the clauses provided — do not invent new ones.
    4. Remember the contract’s **content and purpose**. Their goal is likely not to avoid climate-related risks, but to meet other business or legal needs. We are telling them where they can inject climate-aligned language into the existing contract but the existing contract and its goals are the most important consideration.
    5. Pay close attention to what the contract is **doing** — the transaction type, structure, and key obligations — not just who the parties are or what sector they operate in.
    - Clauses must fit the **actual function and scope** of the contract.
    - For example, do not recommend a clause about land access if the contract is about software licensing.
    - Another example: do not recommend a clause about insurance if the contract is establishing a joint venture.
    6. Consider the relationship between the parties (e.g. supplier–customer, insurer–insured, JV partners).
    - If a clause assumes a different relationship, only suggest it if it can **realistically be adapted**, and explain how.
    7. You may include a clause that is not a perfect match if:
    - It serves a similar **legal or operational function**, and
    - You clearly explain how it could be adapted to the contract context.
    8. Do not recommend clauses that clearly mismatch the contract’s type, scope, or parties.
    9. Avoid redundancy. If the contract already addresses a topic (e.g. dispute resolution), only suggest a clause on that topic if it adds clear value.

    Focus on legal function, contextual fit, and the actual mechanics of the contract. You are recommending **starting points** — plausible clauses the user could adapt.'''

    messages.append({"role": "user", "content": clause_block})

    response = client.chat.completions.create(
        model="deepseek/deepseek-r1-0528-qwen3-8b:free",  
        messages= messages,
        temperature=0.1,
    )

    response_text = response.choices[0].message.content
    df_response = utils.parse_response(response_text)
    
    missing = []
    for clause in df_response["Clause Name"]:
        target = clause + ".txt"
        # try to find at least one close match in your names list
        close = utils.get_close_matches(target, names, n=1, cutoff=0.8)
        if not close:
            missing.append(clause)
            
    if missing:
        print(f"[WARNING] Clauses not found: {missing}")
        # tack on the assistant’s bad output, then our correction prompt
        messages.append({"role":"assistant","content":response_text})
        messages.append({
            "role":"user",
            "content": (
                "One of the clauses you recommended "
                f"({', '.join(missing)}) was not in the provided set. "
                "Do not hallucinate: only pick from the list I gave you, "
                "and please try again."
            )
        })

        # re-call the LLM
        retry = client.chat.completions.create(
            model="deepseek/deepseek-r1-0528-qwen3-8b:free",
            messages=messages,
            temperature=0.1,
        )
        response_text = retry.choices[0].message.content
        df_response = utils.parse_response(response_text)

    return {
        "matches": [
            {
                "name": row["Clause Name"].replace(".txt", ""),
                "reason": row["Reasoning"]
            }
            for _, row in df_response.iterrows()
        ]
    }

def normalize_title(s: str) -> str:
    # 1) remove any parenthetical
    s = re.sub(r"\(.*?\)", " ", s)
    # 2) drop all punctuation
    s = re.sub(r"[^\w\s]", " ", s)
    # 3) lowercase & collapse whitespace
    return re.sub(r"\s+", " ", s.lower()).strip()

_normalized_map = {
    normalize_title(full): full
    for full in names
}

@app.get("/clause/{clause_name}")
def get_clause(clause_name: str):
    raw = unquote(clause_name)
    norm_in = normalize_title(raw)

    # exact normalized-match
    if norm_in in _normalized_map:
        full_title = _normalized_map[norm_in]

    else:
        # substring fallback
        substrs = [full for k, full in _normalized_map.items() if norm_in in k]
        if len(substrs) == 1:
            full_title = substrs[0]
        else:
            # as last resort, fuzzy match on normalized keys
            from difflib import get_close_matches
            close = get_close_matches(norm_in, _normalized_map.keys(), n=1, cutoff=0.5)
            if not close:
                raise HTTPException(404, f"Clause '{raw}' not found")
            full_title = _normalized_map[close[0]]

    idx = names.index(full_title)
    return {"name": full_title, "text": docs[idx]}

# --- Serve Frontend ---
@app.get("/")
def read_root(credentials: HTTPBasicCredentials = Depends(verify_credentials)):
    return FileResponse(INDEX_PATH)

# --- Run with Uvicorn ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("recommender_app:app", host="0.0.0.0", port=8000, reload=True)