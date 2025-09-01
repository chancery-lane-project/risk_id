#imports 
import json
import os
import random
import re
import shutil
import zipfile
from collections import Counter
from difflib import get_close_matches
from typing import List

import hdbscan
import hdbscan.prediction
import numpy as np
import pandas as pd
import plotly.express as px
import torch
import umap.umap_ as umap
from bs4 import BeautifulSoup
from joblib import Parallel, delayed
from rapidfuzz.fuzz import token_set_ratio
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
from sklearn.metrics import (
    pairwise_distances,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from torch.nn.functional import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Detector utils #
#-------------------------------------------------------------------#

CAT0 = "unlikely"
CAT1 = "possible"
CAT2 = "likely"
CAT3 = "very likely"

ALPHA_PATTERN = re.compile(r"[a-zA-Z]")

def load_labeled_contracts(data_folder, modified=False):
    texts = []
    labels = []
    contract_ids = []
    contract_level_labels = []

    contract_level_label = 1 if modified else 0

    for root, _, files in os.walk(data_folder):
        for file in files:
            if not file.endswith(".txt"):
                continue

            contract_path = os.path.join(root, file)
            contract_id = file

            with open(contract_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or len(line) < 36 or not ALPHA_PATTERN.search(line):
                        continue  # Skip short or non-alphabetic lines

                    label = int(line[0])
                    text = line[1:].strip()

                    texts.append(text)
                    labels.append(label)
                    contract_ids.append(contract_id)
                    contract_level_labels.append(contract_level_label)

    return texts, labels, contract_ids, contract_level_labels

def create_and_clean_base_df(texts, labels, contract_ids, contract_level_labels):
    data = pd.DataFrame(
        {
            "contract_ids": contract_ids,
            "text": texts,
            "label": labels,
            "contract_label": contract_level_labels,
        }
    )

    # add a binary column that indicates whether it is a real clause; for all of these, that is 0
    data["real_clause"] = 0
    # remove rows where text is empty
    data = data[data["text"].str.strip().astype(bool)]

    return data

def find_ending_row(data, end_index):
    last_row = data.iloc[end_index]
    if last_row["index"] != data.iloc[end_index + 1]["index"]:
        return end_index
    else:
        return find_ending_row(data, end_index + 1)


def custom_train_test_split(full_data, real_clause_column):
    # setting the real clauses aside for exclusive use in the training set
    real_clauses = full_data[full_data[real_clause_column] == 1]
    rest_data = full_data[full_data[real_clause_column] == 0]
    train_data_temp = real_clauses

    # now that the clauses are remove, randomly shuffle the other data (but keep the contract ids together)
    grouped = list(rest_data.groupby("index"))
    random.shuffle(grouped)
    rest_data = pd.concat([group for name, group in grouped])

    train_size = int(0.75 * len(rest_data))
    val_size = int(0.1 * len(rest_data))
    # don't need the test size because it will be the rest of the data

    train_end = find_ending_row(rest_data, train_size)
    val_end = find_ending_row(rest_data, train_end + val_size)

    train_data = rest_data[:train_end]
    val_data = rest_data[train_end + 1 : val_end]
    test_data = rest_data[val_end + 1 :]

    # add the real clauses back in
    train_data = pd.concat([train_data, train_data_temp])

    # remember the indices of the train, val, and test data
    train_indices = train_data.index.tolist()
    val_indices = val_data.index.tolist()
    test_indices = test_data.index.tolist()

    # print the percentage of data in each split rounded to 2 decimal places and with a % sign
    print("Train: " + str(round(len(train_data) / len(full_data) * 100, 2)) + "%")
    print("Validation: " + str(round(len(val_data) / len(full_data) * 100, 2)) + "%")
    print("Test: " + str(round(len(test_data) / len(full_data) * 100, 2)) + "%")

    return train_data, val_data, test_data, train_indices, val_indices, test_indices

def highlight_climate_content(results_df, text_column="sentence", prediction_column="prediction", keyword_column="contains_climate_keyword"):
    highlighted_text = ""

    for _, row in results_df.iterrows():
        text = row[text_column]
        prediction = row[prediction_column]
        keyword_match = row[keyword_column]

        if prediction == 1 and keyword_match:
            # Highlight in green for prediction + keyword match
            color = "lightgreen"
            highlighted_segment = (
                f"<span style='background-color: {color};'>{text}</span>"
            )
            highlighted_text += highlighted_segment + "<br><br>"

        elif prediction == 1 and not keyword_match:
            # Highlight in yellow for prediction only
            color = "yellow"
            highlighted_segment = (
                f"<span style='background-color: {color};'>{text}</span>"
            )
            highlighted_text += highlighted_segment + "<br><br>"

        else:
            # No highlight
            highlighted_text += text + "<br><br>"

    # Wrap in basic HTML structure
    html_content = f"<html><body>{highlighted_text}</body></html>"
    return html_content


def save_file(filename, content):
    with open(filename, "w") as f:
        f.write(content)


def create_contract_df(results_df, processed_contracts, labelled=True):
    X = results_df['sentence']
    data = processed_contracts 
    y_pred = results_df['prediction']
    keyword_match = results_df['contains_climate_keyword']
    
    X_df = pd.DataFrame({"text": X, "prediction": y_pred, "contains_climate_keyword": keyword_match})
    combined_df = pd.concat([X_df, data], axis=1)

    # Group by contract_id and compute the sum of predicted positive clauses
    contract_level_preds = combined_df.groupby("index")["prediction"].sum()
    contract_level_preds.sort_values(ascending=False, inplace=True)

    # Compute keyword pass/fail per contract
    keyword_pass_dict = {}
    for contract_id, group in combined_df.groupby("index"):
        predicted_clauses = group[group["prediction"] == 1]
        if not predicted_clauses.empty:
            no_keyword_count = (predicted_clauses["contains_climate_keyword"] == False).sum()
            total_predicted = len(predicted_clauses)
            # If 40% or more don't have a keyword match, mark as False
            keyword_pass = (no_keyword_count / total_predicted) < 0.5
        else:
            keyword_pass = True  # If no clauses predicted as 1, we assume pass (or set to False if you prefer)
        keyword_pass_dict[contract_id] = keyword_pass

    keyword_pass_series = pd.Series(keyword_pass_dict)

    if labelled:
        contract_level_labels = combined_df.groupby("index")[
            "contract_label"
        ].first()
        contract_level_df = pd.concat(
            [contract_level_labels, contract_level_preds, keyword_pass_series], axis=1
        )
        contract_level_df.columns = ["contract_label", "prediction", "keyword_pass"]
    else:
        contract_level_df = pd.concat([contract_level_preds, keyword_pass_series], axis=1)
        contract_level_df.columns = ["prediction", "keyword_pass"]

    contract_level_df.reset_index(inplace=True)
    return contract_level_df

def process_text_document(text):
    # Remove spaces around specific punctuation and ensure spacing consistency
    text = re.sub(
        r"\s*([;:\[\]\(\)“”])\s*", r"\1", text
    )  # Remove spaces around these symbols
    text = re.sub(
        r"\s*([,;])", r"\1 ", text
    )  # Ensure a space after commas and semicolons
    text = re.sub(r"\s+", " ", text)  # Normalize multiple spaces to a single space
    text = text.strip()  # Remove leading and trailing whitespace

    # Remove all instances of "[END]"
    text = re.sub(r"\s*\[END\]\s*", "", text)

    # Ensure text ends with a proper punctuation mark
    if text and text[-1] not in {".", "!", "?"}:
        text = text.rstrip(text[-1]) + "."

    # Add paragraph breaks after each period not following specific exceptions
    text = re.sub(
        r"(?<!\d)(?<!\b[A-Z])(?<!\bNo)(?<!\bi\.e)(?<!\be\.g)\. ", ".\n\n", text
    )

    # Remove leading/trailing dashes and replace excessive dashes within the text
    text = re.sub(r"^-+|-+$", "", text, flags=re.MULTILINE)
    text = re.sub(r"-{2,}", "-", text)

    return text

def process_single_contract(file_path, texts, contract_ids):
    """
    Process a single contract file and append text data and contract IDs to the provided lists.
    """
    contract_id = os.path.basename(file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        # Combine all lines from the document
        document = []
        for line in f:
            line = line.strip()
            if line:
                # remove the label if one is present
                if line[0] in {"0", "1"} and line[1:].strip():
                    line = line[1:].strip()
                else:
                    line = line
                # filter lines with no alphabetic characters or shorter than 35 characters
                if len(line) >= 35 and re.search(r"[a-zA-Z]", line):
                    document.append(line)

        # process entire document
        if document:
            full_text = " ".join(document)
            processed_text = process_text_document(full_text)

            split_lines = processed_text.split("\n\n")

            for line in split_lines:
                texts.append(line.strip())
                contract_ids.append(contract_id)

def load_unlabelled_contract(contract_path):
    texts = []
    contract_ids = []

    if os.path.isfile(contract_path):
        process_single_contract(contract_path, texts, contract_ids)
    elif os.path.isdir(contract_path):
        for root, _, files in os.walk(contract_path):
            for file in files:
                if file.endswith(".txt"):
                    full_path = os.path.join(root, file)
                    process_single_contract(full_path, texts, contract_ids)
    else:
        raise ValueError(
            f"Invalid path: {contract_path}. Please provide a valid file or folder path."
        )
    df = pd.DataFrame({"index": contract_ids, "text": texts})
    return df


def create_threshold_buckets(contract_df):
    # Add a temporary column to track adjusted scores
    df = contract_df.copy()

    # Step 1: Determine initial bucket level
    def assign_bucket(score):
        if score >= 7:
            return 3
        elif score >= 3:
            return 2
        elif score >= 1:
            return 1
        else:
            return 0

    df["bucket"] = df["prediction"].apply(assign_bucket)

    # Step 2: Downgrade if keyword_pass == False
    df.loc[df["keyword_pass"] == False, "bucket"] -= 1

    # Step 3: Ensure bucket is not less than 0
    df["bucket"] = df["bucket"].clip(lower=0)

    # Step 4: Assign contracts to buckets
    bucket_0 = df[df["bucket"] == 0]  # none
    bucket_1 = df[df["bucket"] == 1]  # could contain
    bucket_2 = df[df["bucket"] == 2]  # likely
    bucket_3 = df[df["bucket"] == 3]  # very likely

    return bucket_0, bucket_1, bucket_2, bucket_3

def list_all_txt_files(base_dir):
    txt_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".txt"):
                relative_path = os.path.relpath(os.path.join(root, file), base_dir)
                txt_files.append(relative_path)
    return txt_files


def make_folders(category_0, category_1, category_2, category_3, temp_dir, output_folder):
    # Map category names (e.g., "unlikely") to their respective folder paths
    folders = {
        CAT0: os.path.join(output_folder, CAT0),
        CAT1: os.path.join(output_folder, CAT1),
        CAT2: os.path.join(output_folder, CAT2),
        CAT3: os.path.join(output_folder, CAT3),
    }

    # Ensure all folders exist
    for folder in folders.values():
        os.makedirs(folder, exist_ok=True)

    # Create a mapping of base filenames to full relative paths
    uploaded_files = {
        os.path.basename(file): file for file in list_all_txt_files(temp_dir)
    }

    # Map category names to the actual DataFrame objects
    categories = {
        CAT0: category_0,
        CAT1: category_1,
        CAT2: category_2,
        CAT3: category_3,
    }

    # Copy files into their respective folders
    for category_name, contracts_df in categories.items():
        for _, contract in contracts_df.iterrows():
            contract_id = contract["index"]
            if contract_id in uploaded_files:
                source_path = os.path.join(temp_dir, uploaded_files[contract_id])
                destination_path = os.path.join(folders[category_name], contract_id)
                shutil.copy(source_path, destination_path)
            else:
                print(f"File not found for: {contract_id}")

    # Return folder paths in original input order
    return (
        folders[CAT0],
        folders[CAT1],
        folders[CAT2],
        folders[CAT3],
    )

def print_single(category_0, category_1, category_2, category_3, return_result=False):
    category_map = [
        (category_3, CAT3),
        (category_2, CAT2),
        (category_1, CAT1),
        (category_0, CAT0),
    ]

    for df, label in category_map:
        if len(df) != 0:
            return label if return_result else print(label)

    return "unknown" if return_result else print("unknown")
        
def print_percentages(category_0, category_1, category_2, category_3, contract_df, return_result=False):
    total = len(contract_df)

    categories = [
        ("CAT0", category_0),
        ("CAT1", category_1),
        ("CAT2", category_2),
        ("CAT3", category_3),
    ]

    percentages = {}
    for cat_name, group in categories:
        label = globals().get(cat_name, cat_name)
        percentage = round(len(group) / total * 100, 2) if total > 0 else 0.0
        percentages[label] = percentage
        print(f"{label.replace('_', ' ').title()}: {percentage}%")

    return percentages if return_result else None


def zip_folder(folder_path, zip_file_path):
    with zipfile.ZipFile(zip_file_path, "w") as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, folder_path))
                
climate_keywords = [
    "adaptation", "agriculture", "air pollutants", "air quality", "allergen",
    "alternative energy portfolio standard", "animal health", "asthma", "atmosphere",
    "cafe standards", "cap and trade", "cap-and-trade-program", "carbon asset risks",
    "carbon controls", "carbon dioxide", "co2", "carbon footprint", "carbon intensity",
    "carbon pollution", "carbon pollution standard", "carbon tax", "catastrophic events",
    "ch4", "changing precipitation patterns", "clean air act", "clean energy", "clean power plan",
    "climate", "climate change", "climate change regulation", "climate change risk",
    "climate disclosure", "climate issues", "climate opportunities", "climate reporting",
    "climate risk", "climate risk disclosure", "climate risks", "climate-related financial risks",
    "conference of the parties", "corporate average fuel economy", "crop failure", "droughts",
    "earthquakes", "ecosystem", "emission", "emissions", "emissions certificates",
    "emissions trading scheme", "emissions trading system", "emit", "environmental permits",
    "ets", "eu ets", "extinction", "extreme weather", "extreme weather event", "fee and remission",
    "fire season", "flooding", "fossil fuel", "fossil fuel reserves", "fossil fuels", "fuel economy",
    "ghg", "ghg emissions", "ghg regulation", "ghg trades", "global average temperature",
    "global climate", "global warming", "global warming potential", "gwp", "green",
    "green initiatives", "greenhouse effect", "greenhouse gas", "greenhouse gases",
    "gwp source", "habitat", "heat waves", "heavy precipitation", "hfcs", "high temperatures",
    "human health", "hurricanes", "hydro fluorocarbon", "infectious disease", "insured losses",
    "intended nationally determined contribution", "intergovernmental panel on climate change",
    "invasive species", "kyoto protocol", "lcfs", "low carbon fuel standard", "methane",
    "mitigation", "montreal protocol", "n2o", "natural disasters", "natural gas", "nf3",
    "nitrogen oxides", "nox", "nitrogen trifluoride", "nitrous oxide", "oil", "opportunities regulations",
    "ozone", "ozone-depleting substances", "ods", "paris agreement", "paris climate accord",
    "particulate matter", "parts per million", "per fluorocarbons", "pfcs", "persistent organic pollutants",
    "physical risks", "pollutant", "pre-industrial levels of carbon dioxide", "precipitation",
    "precipitation patterns", "rain", "rainfall", "rainwater", "regulation or disclosure of gh emissions",
    "regulatory risks", "renewable", "renewables", "renewable energy", "renewable energy goal",
    "renewable energy standard", "renewable portfolio standard", "renewable resource", "reserves",
    "risks from climate change", "risks regulations", "rps", "sea level rise", "sea-level rise",
    "sf6", "significant air emissions", "solar radiation", "sulfur oxides", "sox",
    "sulphur hexafluoride", "sustainab*", "temperatures", "ultraviolet radiation",
    "ultraviolet (uv-b) radiation", "united nations framework convention on climate change",
    "water availability", "water supply", "water vapor", "weather", "weather events",
    "weather impacts", "wildfires", "energy", "energy efficiency", "energy transition",
]

wildcard_keywords = [kw.replace('*', '.*') for kw in climate_keywords]
regex_patterns = [rf"\b{kw}\b" for kw in wildcard_keywords]
compiled_patterns = [re.compile(p) for p in regex_patterns]

def climate_keyword_matching(sentence: str, patterns: List[re.Pattern], threshold: int = 85) -> bool:
    sentence_lower = sentence.lower()

    # Fast regex pass
    for pattern in patterns:
        if pattern.search(sentence_lower):
            return True

    # Skip fuzzy check if sentence is very short
    if len(sentence_lower.split()) < 4:
        return False

    # Optimized fuzzy match
    for keyword in climate_keywords:
        if token_set_ratio(sentence_lower, keyword.lower()) >= threshold:
            return True

    return False

def add_climate_keyword_column(df: pd.DataFrame, text_column: str = "sentence") -> pd.DataFrame:
    df = df.copy()
    df["contains_climate_keyword"] = Parallel(n_jobs=-1)(
        delayed(climate_keyword_matching)(x, compiled_patterns) for x in df[text_column]
    )
    return df

def create_result_df(results, processed_contracts): 
    result_df = pd.DataFrame({
    "prediction": results,
    "sentence": processed_contracts["text"]
    })
    
    result_df = add_climate_keyword_column(result_df)
    
    result_df_true = result_df[result_df['contains_climate_keyword'] == True]
    
    return result_df, result_df_true

def predict_climatebert(texts, tokenizer, device, model, batch_size=16):
    model.eval()  # Ensure eval mode

    all_preds = []
    all_probs = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]

        # Tokenize on CPU then move each tensor to device
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        )
        inputs = {key: val.to(device) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probs = softmax(outputs.logits, dim=1)
            preds = torch.argmax(probs, dim=1)

        all_preds.append(preds.cpu())
        all_probs.append(probs.cpu())

    # Concatenate tensors once (faster than extend + numpy conversion per loop)
    all_preds = torch.cat(all_preds).numpy()
    all_probs = torch.cat(all_probs).numpy()

    return all_preds, all_probs

# Recommender utils#
#-------------------------------------------------------------------#
# utils.py
"""This is the utils file for the clause_recommender task."""
    
def load_clauses(clause_folder, childs_name=False):
    documents = []
    file_names = []
    clause_titles = []
    child_names = []

    for fname in sorted(os.listdir(clause_folder)):
        if fname.endswith(".txt"):
            path = os.path.join(clause_folder, fname)
            with open(path, "r") as f:
                content = f.read()
                documents.append(content)
                file_names.append(fname)

                # Parse HTML to extract <h4> title
                soup = BeautifulSoup(content, "html.parser")
                title_tag = soup.find("h4")
                if title_tag:
                    clause_titles.append(title_tag.text.strip())
                else:
                    clause_titles.append(fname.replace(".txt", ""))  # fallback

                # Optionally extract child's name
                if childs_name:
                    child_tag = soup.find("p", class_="childs-name")
                    if child_tag:
                        child_names.append(child_tag.text.strip())
                    else:
                        child_names.append("")  # fallback if not present

    if childs_name:
        return documents, file_names, clause_titles, child_names
    else:
        return documents, file_names, clause_titles

def custom_stop_words():
    legal_words = [
        "clause",
        "agreement",
        "contract",
        "parties",
        "shall",
        "herein",
        "company",
        "date",
        "form",
        "party",
        "ii",
        "pursuant",
        "document",
        "term",
        "condition",
        "obligation",
        "rights",
        "include",
        "including",
        "one",
        "such",
        "set",
        "environment",
        "environmental",
        "jurisdiction",
        "jurisdictional",
        "class",
        "clause",
        "clauses"
    ]
    # Ensure all words are lowercase
    combined = ENGLISH_STOP_WORDS.union(w.lower() for w in legal_words)
    return list(combined)

def get_matching_clause(query_vector, document_vectors, clause_names):
    # Compute cosine similarities using a single call
    cosine_similarities = cosine_similarity(query_vector, document_vectors).flatten()

    # Retrieve the best match and its details
    most_similar_index = np.argmax(cosine_similarities)
    most_similar_score = cosine_similarities[most_similar_index]
    similarity_range = cosine_similarities.max() - cosine_similarities.min()
    most_similar_clause = clause_names[most_similar_index]

    return (
        most_similar_clause,
        most_similar_score,
        most_similar_index,
        cosine_similarities,
        similarity_range,
    )

def find_top_k(similarities, clause_names, k=3):
    best_match_indices = np.argsort(similarities)[::-1][:k]
    best_match_names = [clause_names[i] for i in best_match_indices]
    best_match_scores = [similarities[i] for i in best_match_indices]
    return best_match_names, best_match_scores, similarities

def output_feature_chart(vectorizer, X, most_similar_index):
    # Get the feature names (words) from the vectorizer
    feature_names = vectorizer.get_feature_names_out()

    # Convert the document-term matrix to an array and isolate the most similar document and target doc
    X_array = X.toarray()
    target_vector = X_array[-1]  # Target document vector
    most_similar_vector = X_array[most_similar_index]  # Most similar document vector

    # Create DataFrames to easily view word frequencies
    target_df = pd.DataFrame({"word": feature_names, "target_frequency": target_vector})
    similar_df = pd.DataFrame(
        {"word": feature_names, "similar_frequency": most_similar_vector}
    )

    # Merge on words and filter for words that have non-zero counts in both documents
    merged_df = target_df.merge(similar_df, on="word")
    merged_df = merged_df[
        (merged_df["target_frequency"] > 0) & (merged_df["similar_frequency"] > 0)
    ]

    # Sort by combined frequency (or just display both frequencies)
    merged_df["total_frequency"] = (
        merged_df["target_frequency"] + merged_df["similar_frequency"]
    )
    merged_df = merged_df.sort_values(by="total_frequency", ascending=False)

    return merged_df

def cls_pooling(outputs):
    return outputs.last_hidden_state[:, 0, :].cpu().numpy()


def encode_text(text, tokenizer, model, token_index=None):
    inputs = tokenizer(
        text, return_tensors="pt", padding=True, truncation=True, max_length=512
    )
    with torch.no_grad():
        outputs = model(**inputs)
    return cls_pooling(outputs)


def find_top_similar_bow(target_doc, documents, file_names, similarity_threshold=0.15, k=30):
    custom_stop_words_list = custom_stop_words()
    vectorizer = CountVectorizer(stop_words=custom_stop_words_list, token_pattern=r"(?u)\b[a-zA-Z]{2,}\b")
    all_documents = documents + [target_doc]
    X = vectorizer.fit_transform(all_documents)
    document_vectors = X[:-1]
    query_vector = X[-1:]

    # Call helper method to get the most similar clause
    (
        _,
        _,
        most_similar_index,
        similarities,
        _,
    ) = get_matching_clause(query_vector, document_vectors, file_names)

    # Find the top three matching clauses
    best_match_names, best_match_scores, similarities = find_top_k(
        similarities, file_names, k = k
    )
    merged_df = output_feature_chart(vectorizer, X, most_similar_index)

    return {
    "Top_Matches": best_match_names,
    "Scores": best_match_scores,
    "Documents": [documents[file_names.index(name)] for name in best_match_names],
    "Feature_Chart": merged_df,
    }
    
def get_embedding_matches_subset(
    query_text, 
    documents_subset, 
    names_subset, 
    tokenizer, 
    model, 
    k =3
):
    # Embed the top-K candidate clauses
    subset_embeddings = np.vstack([
        encode_text(doc, tokenizer, model)
        for doc in documents_subset
    ])
    # Embed the query
    query_embedding = encode_text(query_text, tokenizer, model).reshape(1, -1)

    # Compute cosine similarity
    _, _, _, similarities, _ = get_matching_clause(query_embedding, subset_embeddings, names_subset)

    # Get top 3 (or however many you want)
    top_names, top_scores, _ = find_top_k(similarities, names_subset, k=k)
    top_texts = [documents_subset[names_subset.index(name)] for name in top_names]

    return top_names, top_scores, top_texts

def parse_clause_boxes_to_df(clause_boxes):
    data = []

    for box in clause_boxes:
        soup = BeautifulSoup(box, "html.parser")
        name = soup.find("p", class_="childs-name").get_text(strip=True) if soup.find("p", class_="childs-name") else None
        title = soup.find("h4").get_text(strip=True) if soup.find("h4") else None
        excerpt = soup.find("p", class_="excerpt").get_text(strip=True) if soup.find("p", class_="excerpt") else None

        # Initialize
        jurisdiction, updated = None, None
        
        meta_data = soup.find_all("p", class_="meta-data")
        for md in meta_data:
            leadin = md.find("span", class_="cfc-leadin")
            value = md.find("span", class_="cfc-taxonomy")
            if leadin and value:
                leadin_text = leadin.get_text(strip=True)
                value_text = value.get_text(strip=True)
                if "Jurisdiction" in leadin_text:
                    jurisdiction = value_text
                elif "Updated" in leadin_text:
                    updated = value_text

        url_tag = soup.find("a", class_="hot-spot")
        url = url_tag['href'] if url_tag and url_tag.has_attr('href') else None

        data.append({
            "Name": name,
            "Title": title,
            "Excerpt": excerpt,
            "Jurisdiction": jurisdiction,
            "Updated": updated,
            "URL": url
        })

    return pd.DataFrame(data)

def clean_string(s):
    s = s.lower()
    s = re.sub(r'[\W_]+', '_', s)  # Replace non-alphanumeric characters with underscores
    s = re.sub(r'_+', '_', s)  # Collapse multiple underscores
    return s.strip('_')

def attach_documents(df, documents, file_names):
    # Preprocess file_names
    file_map = {clean_string(name.replace('.txt', '')): (name, doc) for name, doc in zip(file_names, documents)}
    
    records = []
    missing_titles = []
    
    for idx, row in df.iterrows():
        title = row['Title']
        cleaned_title = clean_string(title)
        possible_matches = get_close_matches(cleaned_title, file_map.keys(), n=1, cutoff=0.6)
        
        if possible_matches:
            best_match = possible_matches[0]
            filename, text = file_map[best_match]
            new_row = row.copy()
            new_row['Matched Filename'] = filename
            new_row['Document'] = text
            records.append(new_row)
        else:
            missing_titles.append(title)
    
    # Build new df only with matched records
    result_df = pd.DataFrame(records)
    
    if missing_titles:
        print("\n⚠️ Titles with no matching document:")
        for title in missing_titles:
            print(f" - {title}")
    
    return result_df
        
def rebuild_documents(df):
    filenames = []
    documents = []
    
    for idx, row in df.iterrows():
        filename = str(row['Title']).strip()  # Force safe string
        excerpt = str(row['Excerpt'] or "").strip()
        document = str(row['Document'] or "").strip()
        
        combined_text = f"{excerpt}\n\n{document}".strip()
        
        filenames.append(filename)
        documents.append(combined_text)
    
    return filenames, documents

def create_name_to_child_mapping(final_df):
    """
    Create a mapping from clause names (titles) to child names.
    """
    name_to_child = {}
    for _, row in final_df.iterrows():
        title = row['Title']
        child_name = row['Name'] if pd.notna(row['Name']) else ""
        name_to_child[title] = child_name
    return name_to_child

def getting_started(model_path, clause_folder, clause_html):
    model_path = os.path.abspath(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    detector_model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
    classifier_model = detector_model.base_model
    # this is specific to embeddings; we have now lost the classification head

    documents, file_names, _ = load_clauses(clause_folder)

    # Get child's name from clause_html
    clause_boxes, _, _, child_names = load_clauses(clause_html, childs_name=True)
    clause_box_df = parse_clause_boxes_to_df(clause_boxes)
    final_df = attach_documents(clause_box_df, documents, file_names)
    names, docs = rebuild_documents(final_df)
    
    # Create a mapping from clause names to child names
    name_to_child = create_name_to_child_mapping(final_df)
    
    return tokenizer, detector_model, classifier_model, names, docs, final_df, child_names, name_to_child

def combine_title_and_text(row, title_to_document):
    title = row['Clause']
    cleaned_title = clean_string(title)
    
    body = title_to_document.get(cleaned_title)
    if body:
        return f"Title: {title}\n\nText: {body}"
    
    # If direct lookup failed, try fuzzy matching
    possible_matches = get_close_matches(cleaned_title, title_to_document.keys(), n=1, cutoff=0.6)
    if possible_matches:
        best_match = possible_matches[0]
        print(f"Fuzzy match: '{title}' matched to '{best_match}'")
        body = title_to_document[best_match]
        return f"Title: {title}\n\nText: {body}"
    else:
        print(f"No match found for title: '{title}' (cleaned: '{cleaned_title}')")
        return f"Title: {title}\n\nText: [NO MATCH FOUND]"
    
def most_common_tag(tags):
    return Counter(tags).most_common(1)[0][0]

def multi_label_jacccard(clause_tags, visualize = False):
    mlb = MultiLabelBinarizer()
    tag_matrix = mlb.fit_transform(clause_tags['Tag'])

    # Jaccard distance: 1 - (intersection / union)
    jaccard_distances = pairwise_distances(tag_matrix, metric='jaccard')
    umap_jaccard = umap.UMAP(metric='precomputed', random_state=42)
    jaccard_2d = umap_jaccard.fit_transform(jaccard_distances)
    
    if visualize:
        plot_df = pd.DataFrame({
            "x": jaccard_2d[:, 0],
            "y": jaccard_2d[:, 1],
            "NumTags": clause_tags['Tag'].apply(len),
            "PrimaryTag": clause_tags['PrimaryTag'],
            "Clause": clause_tags['Clause'],
            "Hover": clause_tags.apply(lambda row: f"{row['Clause'][:100]}<br>Tags: {', '.join(row['Tag'])}", axis=1)
        })

        fig = px.scatter(
            plot_df,
            x="x",
            y="y",
            color="PrimaryTag",  # or use "NumTags" if you want a heat-style gradient
            hover_name="Clause",
            hover_data={"Hover": True},
            title="Jaccard UMAP of TCLP Clauses by Tag Overlap"
        )
        fig.update_traces(marker=dict(size=6, opacity=0.7))
        fig.show()
    
    return tag_matrix

def perform_hdbscan(tag_matrix, embeddings): 
    hybrid_features = np.hstack([embeddings, tag_matrix])

    umap_model = umap.UMAP(random_state=42)
    hybrid_2d = umap_model.fit_transform(hybrid_features)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=5,
        min_samples=1,
        cluster_selection_epsilon=0.6,
        prediction_data=True
    )
    _ = clusterer.fit_predict(hybrid_2d)

    soft_labels = hdbscan.prediction.all_points_membership_vectors(clusterer)
    forced_labels = soft_labels.argmax(axis=1)
    
    return forced_labels, hybrid_2d, umap_model

def plot_clusters(clause_tags, hybrid_2d):
    fig = px.scatter(
        x=hybrid_2d[:, 0],
        y=hybrid_2d[:, 1],
        color=clause_tags['HybridCluster'].astype(str),
        hover_data={
            "Clause": clause_tags['Clause'],
            "PrimaryTag": clause_tags['PrimaryTag'],
            "Tags": clause_tags['Tag']
        },
        title="Hybrid UMAP: Text Embeddings + Tags + HDBSCAN Clusters"
    )
    fig.update_traces(marker=dict(size=6, opacity=0.7))
    #make it more square 
    fig.update_layout(
    width=900,
    height=900,
    margin=dict(l=20, r=20, t=40, b=20),
    xaxis=dict(scaleanchor=None),  
    )
    fig.update_yaxes(range=[-5, 5])
    fig.show()
    
def per_cluster_split(X, y, cluster_labels, test_size=0.2, min_test_per_cluster=2, random_state=42):
    """
    Perform train/test split within each cluster to guarantee at least `min_test_per_cluster` from each cluster in the test set.
    """
    X = np.array(X)
    print(X.shape)  # Should be (n_samples, n_features)

    y = np.array(y)
    cluster_labels = np.array(cluster_labels)

    X_train_list, X_test_list = [], []
    y_train_list, y_test_list = [], []

    for label in np.unique(cluster_labels):
        cluster_mask = cluster_labels == label
        X_cluster = X[cluster_mask]
        y_cluster = y[cluster_mask]
        n = len(X_cluster)

        # Determine test size for this cluster
        n_test = max(min_test_per_cluster, int(np.floor(test_size * n)))
        n_test = min(n - 1, n_test)  # Ensure at least one sample remains in train

        # Do a deterministic split
        X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
            X_cluster, y_cluster,
            test_size=n_test,
            random_state=random_state,
            shuffle=True  # You want some randomness but reproducible
        )

        X_train_list.append(X_train_c)
        X_test_list.append(X_test_c)
        y_train_list.append(y_train_c)
        y_test_list.append(y_test_c)

    # Combine per-cluster splits
    X_train = np.vstack(X_train_list)
    X_test = np.vstack(X_test_list)
    y_train = np.hstack(y_train_list)
    y_test = np.hstack(y_test_list)

    return X_train, X_test, y_train, y_test

def perform_cluster(cluster_model, query_embedding, tokenizer, embedding_model, clause_tags, fitted_umap, embed=False):
    # Embed the query if requested
    if embed:
        query_embedding = encode_text(query_embedding, tokenizer, embedding_model)

    query_embedding_umap = fitted_umap.transform(query_embedding.reshape(1, -1))

    # Predict cluster
    pred_cluster = cluster_model.predict(query_embedding_umap)[0]

    # Filter clauses to predicted cluster
    cluster_subset_df = clause_tags[clause_tags['HybridCluster'] == pred_cluster]

    # Extract text and names
    subset_docs = cluster_subset_df['CombinedText'].tolist()
    subset_names = cluster_subset_df['Clause'].tolist()
    
    return subset_docs, subset_names, cluster_subset_df

def prepare_clause_tags(clause_tags, final_df): 
    cleaned_titles = final_df['Title'].apply(clean_string)
    title_to_document = dict(zip(cleaned_titles, final_df['Document']))
    
    clause_tags["CombinedText"] = clause_tags.apply(
    lambda row: combine_title_and_text(row, title_to_document), axis=1
    )
    clause_tags['Tag'] = clause_tags['Tag'].apply(lambda x: [tag.strip() for tag in x.split(',')])
    
    clause_tags['PrimaryTag'] = clause_tags['Tag'].apply(most_common_tag)

    return clause_tags

def parse_response(response_text):
    """
    Parses a JSON-formatted LLM response, optionally wrapped in Markdown code fences.
    Returns a list of clauses with 'Clause Name' and 'Reasoning'.
    """
    if not response_text.strip():
        print("Empty response text received")
        return None

    # Remove surrounding triple backticks or Markdown code fences if present
    response_text = response_text.strip()
    if response_text.startswith("```"):
        response_text = re.sub(r"^```(?:json)?\s*", "", response_text)
        response_text = re.sub(r"\s*```$", "", response_text)

    try:
        clauses = json.loads(response_text)
        if not isinstance(clauses, list):
            print(f"Expected a list of clause dictionaries, got: {type(clauses)}")
            return None
        df = pd.DataFrame(clauses)
        for c in clauses:
            if not all(k in c for k in ["Clause Name", "Reasoning"]):
                print(f"Missing expected keys in clause object: {c}")
                return None
        return df[["Clause Name", "Reasoning"]]
    except json.JSONDecodeError as e:
        print(f"Invalid JSON format: {e}")
        print(f"Response text: {response_text}")
        return None
    except Exception as e:
        print(f"Unexpected error parsing response: {e}")
        return None


# Risk utils#
#-------------------------------------------------------------------#

def format_taxonomy_prompt(risk_taxonomy, given_prompt):
    prompt = given_prompt 
    prompt += "Here are the available categories:\n\n"
    for _, row in risk_taxonomy.iterrows():
        prompt += f"- `{row['Label']}`: {row['Description']}\n"
    prompt += "\n"
    prompt += "Return only the label that best applies, and explain your reasoning.\n"
    return prompt

def classify_clause(clause_text, taxonomy_df, given_prompt, client, model="qwen/qwen-2.5-7b-instruct"):
    system_prompt = format_taxonomy_prompt(taxonomy_df, given_prompt)
    
    user_prompt = f"""Clause:
        \"\"\"{clause_text}\"\"\"

        Which risk categories does this clause help mitigate?
        Respond in this JSON format:
        {{
        "labels": ["label1", "label2", ...],
        "justification": "Explain why these labels apply to this clause."
        }}"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2
    )
    
    return response.choices[0].message.content

def format_classification_result(name, result_json, risk_labels):
    import json

    # Parse the result if needed
    if isinstance(result_json, str):
        try:
            parsed = json.loads(result_json)
        except json.JSONDecodeError:
            print(result_json)
            parsed = {"labels": [], "justification": "Invalid JSON response"}

    else:
        parsed = result_json

    labels = parsed.get("labels", [])
    justification = parsed.get("justification", "")

    # Start building the row: default to 0 for all risks
    row = {label: "0" for label in risk_labels}
    row["name"] = name
    row["justification"] = justification

    for label in labels:
        if label in risk_labels:
            row[label] = "1"  # mark as positive

    return row

def get_risk_label(response_df, risk_df):
    for name in response_df['Clause Name']:
        #find the name in risk_df 
        if name in risk_df['Title'].values:
            risk_label = risk_df[risk_df['Title'] == name]['combined_labels'].values[0]
            #add this to the response_df
            response_df.loc[response_df['Clause Name'] == name, 'combined_labels'] = risk_label
            print(f"Clause '{name}' found in risk categories with label: {risk_label}.")
        else:
            print(f"Clause '{name}' not found in risk categories.")
            # Set a default value to avoid NaN
            response_df.loc[response_df['Clause Name'] == name, 'combined_labels'] = "No specific risks identified"
            
    return response_df

def get_emission_label(response_df, emission_df):
    for name in response_df['Clause Name']:
        # Find all rows in emission_df that match this clause name
        matches = emission_df[emission_df['name'] == name]
        if not matches.empty:
            # Build a list of dicts for each match
            emission_info = []
            for _, row in matches.iterrows():
                info = {
                    "emission_label": row.get("emission_label", ""),
                    "topic_name": row.get("topic_name", ""),
                    "justification": row.get("justification", "")
                }
                emission_info.append(info)
            # Convert the list to a JSON string for pandas storage
            import json
            response_df.loc[response_df['Clause Name'] == name, 'combined_labels'] = json.dumps(emission_info)
            print(f"Clause '{name}' found in emission categories with labels: {[e['emission_label'] for e in emission_info]}.")
        else:
            print(f"Clause '{name}' not found in emission categories.")
            response_df.loc[response_df['Clause Name'] == name, 'combined_labels'] = json.dumps([{
                "emission_label": "No specific emissions sources identified",
                "topic_name": "",
                "justification": ""
            }])
    return response_df