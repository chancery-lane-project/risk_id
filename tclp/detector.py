import os
import shutil

import utils as du
from fastapi import Depends, FastAPI, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from starlette.status import HTTP_401_UNAUTHORIZED  # Import the status code

app = FastAPI()
MAX_FILE_LIMIT = 1000

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
model_name = os.path.join(
    BASE_DIR, "/app/tclp/clause_detector/clause_identifier_model.pkl"
)
model = du.load_model(model_name)


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

            processed_contracts = du.load_unlabelled_contract(file_path)

        # Model predictions
        results = model.predict(processed_contracts["text"])
        if is_folder == "false":
            highlighted_output = du.highlight_climate_content(
                processed_contracts["text"], results
            )
            du.save_file("highlighted_output.html", highlighted_output)

        contract_df = du.create_contract_df(
            processed_contracts["text"], processed_contracts, results, labelled=False
        )

        likely, very_likely, extremely_likely, none = du.create_threshold_buckets(
            contract_df
        )

        bucket_details = {
            "likely": {
                "count": len(likely),
                "documents": likely["contract_ids"].tolist(),
            },
            "very_likely": {
                "count": len(very_likely),
                "documents": very_likely["contract_ids"].tolist(),
            },
            "extremely_likely": {
                "count": len(extremely_likely),
                "documents": extremely_likely["contract_ids"].tolist(),
            },
            "none": {
                "count": len(none),
                "documents": none["contract_ids"].tolist(),
            },
        }

        if is_folder == "true":
            percentages = du.print_percentages(
                likely,
                very_likely,
                extremely_likely,
                none,
                contract_df,
                return_result=True,
            )
            uploaded_files = os.listdir(temp_dir)
            print("Uploaded Files:", uploaded_files)
            (
                likely_folder,
                very_likely_folder,
                extremely_likely_folder,
                none_folder,
            ) = du.make_folders(
                likely, very_likely, extremely_likely, none, temp_dir, output_dir
            )
            zip_files = {}
            for category, folder in zip(
                ["likely", "very_likely", "extremely_likely", "none"],
                [
                    likely_folder,
                    very_likely_folder,
                    extremely_likely_folder,
                    none_folder,
                ],
            ):
                zip_path = f"{folder}.zip"
                du.zip_folder(folder, zip_path)
                zip_files[category] = zip_path

            # Return download links
            response = {
                "percentages": percentages,
                "buckets": bucket_details,
                "download_links": {
                    "likely_zip": "/output/likely.zip",
                    "very_likely_zip": "/output/very_likely.zip",
                    "extremely_likely_zip": "/output/extremely_likely.zip",
                    "none_zip": "/output/none.zip",
                },
            }
            return JSONResponse(content=response)
        else:
            result = du.print_single(
                likely, very_likely, extremely_likely, none, return_result=True
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
        return JSONResponse(
            content={"error": f"An error occurred: {str(e)}"}, status_code=500
        )


@app.get("/")
def read_root(credentials: HTTPBasicCredentials = Depends(verify_credentials)):
    return FileResponse(os.path.join(BASE_DIR, "/app/tclp/clause_detector/index.html"))


@app.get("/test-file")
async def test_file():
    test_path = os.path.join(output_dir, "test.txt")
    if not os.path.exists(test_path):
        with open(test_path, "w") as f:
            f.write("This is a test file.")
    return FileResponse(test_path)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("detector:app", host="0.0.0.0", port=8080, reload=True)