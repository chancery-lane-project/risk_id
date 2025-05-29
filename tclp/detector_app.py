import os
import shutil
import traceback

import torch
import utils as du
from fastapi import Depends, FastAPI, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from starlette.status import HTTP_401_UNAUTHORIZED  # Import the status code
from transformers import AutoModelForSequenceClassification, AutoTokenizer

app = FastAPI()
MAX_FILE_LIMIT = 1000
CAT0 = "unlikely"
CAT1 = "possible"
CAT2 = "likely"
CAT3 = "very likely"

# Enable CORS for frontend-backend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBasic()

# Dummy credentials for demonstration purposes
USERNAME = "father"
PASSWORD = "christmas"

def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = credentials.username == USERNAME
    correct_password = credentials.password == PASSWORD
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )

# Absolute paths for directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
temp_dir = os.path.join(BASE_DIR, "temp_uploads")
output_dir = os.path.join(temp_dir, "output")

os.makedirs(output_dir, exist_ok=True)
app.mount("/output", StaticFiles(directory=output_dir), name="output")

# Load model when application starts
model_path = os.path.join(BASE_DIR, "models", "CC_BERT", "CC_model_detect")
device = torch.device("cpu")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.to(device)
model.eval()


@app.post("/process/")
async def process_contract(files: list[UploadFile], is_folder: str = Form("false")):
    """
    Endpoint to process a contract file or folder.
    """
    if len(files) > MAX_FILE_LIMIT:
        return JSONResponse(
            content={
                "error": f"This server can only handle up to {MAX_FILE_LIMIT} files; please try again."
            },
            status_code=400,  # Bad Request
        )

    try:
        # Cleanup and recreate temp directories
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        if is_folder == "true":
            print("Processing folder upload...")
            file_paths = []

            for file in files:
                if not file.filename.endswith(".txt"):
                    print(f"Skipping non-txt file: {file.filename}")
                    continue

                file_path = os.path.join(temp_dir, file.filename)
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, "wb") as f:
                    f.write(await file.read())
                file_paths.append(file_path)
                print(f"Stored file: {file_path}")

            if not file_paths:
                return JSONResponse(
                    content={
                        "error": "No valid .txt files found in the uploaded folder."
                    },
                    status_code=400,
                )
        
            processed_contracts = du.load_unlabelled_contract(temp_dir)
            texts = processed_contracts["text"].tolist()

        else:
            print("Processing single file upload...")

            # Handle single file upload
            file = files[0]
            if not file.filename.endswith(".txt"):
                return JSONResponse(
                    content={
                        "error": "Only .txt files are supported for single file uploads."
                    },
                    status_code=400,
                )

            file_path = os.path.join(temp_dir, file.filename)
            with open(file_path, "wb") as f:
                f.write(await file.read())

            processed_contracts = du.load_unlabelled_contract(temp_dir)
            texts = processed_contracts["text"].tolist()

        # Model predictions
        results, _ = du.predict_climatebert(texts, tokenizer, device, model)
        result_df, _ = du.create_result_df(results, processed_contracts)
        
        
        if is_folder == "false":
            highlighted_output = du.highlight_climate_content(result_df)
            du.save_file("highlighted_output.html", highlighted_output)

        contract_df = du.create_contract_df(
            result_df, processed_contracts, labelled=False
        )

        zero, one, two, three = du.create_threshold_buckets(contract_df)
        
        print(CAT0, zero.columns)

        bucket_details = {
            CAT0: {
                "count": len(zero),
                "documents": zero["index"].tolist(),
            },
            CAT1: {
                "count": len(one),
                "documents": one["index"].tolist(),
            },
            CAT2: {
                "count": len(two),
                "documents": two["index"].tolist(),
            },
            CAT3: {
                "count": len(three),
                "documents": three["index"].tolist(),
            },
        }

        if is_folder == "true":
            percentages = du.print_percentages(
                zero,
                one,
                two,
                three,
                contract_df,
                return_result=True,
            )
            uploaded_files = os.listdir(temp_dir)
            print("Uploaded Files:", uploaded_files)
            (
                zero_folder,
                one_folder,
                two_folder,
                three_folder,
            ) = du.make_folders(
                zero, one, two, three, temp_dir, output_dir
            )
            zip_files = {}
            for category, folder in zip(
                [CAT0, CAT1, CAT2, CAT3],
                [
                    zero_folder,
                    one_folder,
                    two_folder,
                    three_folder,
                ],
            ):
                zip_path = f"{folder}.zip"
                du.zip_folder(folder, zip_path)
                zip_files[category] = zip_path

            # Return download links
            response = {
                "percentages": percentages,
                "buckets": bucket_details,
                "bucket_labels": {
                    "cat0": CAT0,
                    "cat1": CAT1,
                    "cat2": CAT2,
                    "cat3": CAT3,
                },
                "download_links": {
                    f"{cat}_zip": f"/output/{os.path.basename(zip_files[cat])}" for cat in [CAT0, CAT1, CAT2, CAT3]
                },
            }
            return JSONResponse(content=response)
        else:
            result = du.print_single(
                zero, one, two, three, return_result=True
            )
            response = {
                "classification": result,
                "highlighted_content": highlighted_output,
            }

        print(response)

        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
        os.makedirs(
            output_dir, exist_ok=True
        )  # Recreate the output directory after cleanup
        return JSONResponse(content=response)

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            content={"error": f"An error occurred: {str(e)}"}, status_code=500
        )


@app.get("/")
def read_root(credentials: HTTPBasicCredentials = Depends(verify_credentials)):
    index_path = os.path.join(BASE_DIR, "detector_index.html")
    return FileResponse(index_path)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("detector_app:app", host="0.0.0.0", port=8080, reload=True)