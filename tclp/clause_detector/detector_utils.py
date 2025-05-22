import os
import pickle
import random
import re
import shutil
import zipfile
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import torch
from joblib import Parallel, delayed
from rapidfuzz.fuzz import token_set_ratio
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.nn.functional import softmax

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


def load_clauses(data_folder, real=True):
    clause_texts = []
    clause_labels = []
    clause_ids = []
    clause_reality = []

    for root, _, files in os.walk(data_folder):
        for file in files:
            if file.endswith(".txt"):
                clause_path = os.path.join(root, file)
                clause_id = file  # Use the file name as an identifier for each clause

                with open(clause_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            # Label every line in clause files as '1'
                            clause_labels.append(1)
                            clause_texts.append(line)
                            clause_ids.append(clause_id)
                            clause_reality.append(1 if real else 0)

    return clause_texts, clause_labels, clause_ids, clause_reality


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


# helper function for train test split so that contracts are kept together
def find_ending_row(data, end_index):
    last_row = data.iloc[end_index]
    if last_row["contract_ids"] != data.iloc[end_index + 1]["contract_ids"]:
        return end_index
    else:
        return find_ending_row(data, end_index + 1)


def custom_train_test_split(full_data, real_clause_column):
    # setting the real clauses aside for exclusive use in the training set
    real_clauses = full_data[full_data[real_clause_column] == 1]
    rest_data = full_data[full_data[real_clause_column] == 0]
    train_data_temp = real_clauses

    # now that the clauses are remove, randomly shuffle the other data (but keep the contract ids together)
    grouped = list(rest_data.groupby("contract_ids"))
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


def save_test_data(test_data, all_contract_path, test_contract_path):
    # save test contracts to folder
    os.makedirs(test_contract_path, exist_ok=True)

    # Extract unique test contract IDs
    test_contracts = test_data["contract_ids"].unique()

    # List all contract files in the source directory
    all_contracts = os.listdir(all_contract_path)

    # Copy each matching contract to the test folder
    for contract in all_contracts:
        if contract in test_contracts:
            source_path = os.path.join(all_contract_path, contract)
            destination_path = os.path.join(test_contract_path, contract)
            shutil.copy(source_path, destination_path)

    print(f"Test contracts have been saved to: {test_contract_path}")


def X_y_split(data, text_column="text", label_column="label"):
    X = data[text_column]
    y = data[label_column]
    return X, y


def save_model(model, model_name):
    with open(model_name, "wb") as f:
        pickle.dump(model, f)


def load_model(model_name):
    with open(model_name, "rb") as f:
        model = pickle.load(f)
    return model


def evaluate_model_clause_level(model, X_val, y_val, verbose=False):
    y_pred = model.predict(X_val)
    print("Classification Report:\n", classification_report(y_val, y_pred))
    print("Accuracy Score:", accuracy_score(y_val, y_pred))
    print("F1 Score:", f1_score(y_val, y_pred))

    if verbose:
        # print misclassified examples
        counter = 0
        for i in range(len(y_pred)):
            if y_pred[i] != y_val.iloc[i]:
                print(X_val.iloc[i])
                print("prediction:", y_pred[i], "actual:", y_val.iloc[i])
                print("\n")
                counter += 1

        print("total misclassified sentences:", counter)

    return y_pred


def threshold_graphs(contract_df, thresholds=range(1, 7), metric_type="f1"):
    # true labels and predictions
    contract_level_true = contract_df["contract_label"]
    contract_level_preds = contract_df["prediction"]

    values = []

    for threshold in thresholds:
        contract_level_preds_threshold = (contract_level_preds >= threshold).astype(int)
        if metric_type == "f1":
            value = f1_score(
                contract_level_true, contract_level_preds_threshold, zero_division=1
            )
        elif metric_type == "accuracy":
            value = accuracy_score(contract_level_true, contract_level_preds_threshold)
        else:
            raise ValueError("Invalid metric_type. Choose 'f1' or 'accuracy'.")

        values.append(value)

    # Plotting threshold vs metric
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, values, marker="o", label=metric_type.capitalize())
    plt.xlabel("Threshold")
    plt.ylabel(metric_type.capitalize())
    plt.title(f"{metric_type.capitalize()} vs Threshold")
    plt.grid(True)
    plt.legend()
    plt.show()

    return values, thresholds


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
    contract_level_preds = combined_df.groupby("contract_ids")["prediction"].sum()
    contract_level_preds.sort_values(ascending=False, inplace=True)

    # Compute keyword pass/fail per contract
    keyword_pass_dict = {}
    for contract_id, group in combined_df.groupby("contract_ids"):
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
        contract_level_labels = combined_df.groupby("contract_ids")[
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


def print_contract_classification_report(df, values, threshold, custom_threshold=None):
    best_threshold_index = values.index(max(values))
    if custom_threshold:
        best_threshold_index = custom_threshold
    best_threshold = int(threshold[best_threshold_index])
    if best_threshold == 0:
        best_threshold = 1
    best_f1 = values[best_threshold_index]
    contract_level_true = df["contract_label"]
    contract_level_preds = df["prediction"]
    best_threshold_preds = (contract_level_preds >= best_threshold).astype(int)
    accuracy = accuracy_score(contract_level_true, best_threshold_preds)
    print(f"Best Threshold: {best_threshold}")
    print("\nBest Threshold Contract-level Classification Report:")
    print(classification_report(contract_level_true, best_threshold_preds))
    print(f"Best Threshold Contract-level F1 Score: {best_f1:.4f}")
    print(f"Best Threshold Contract-level Accuracy Score: {accuracy:.4f}")


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
    df = pd.DataFrame({"contract_ids": contract_ids, "text": texts})
    return df


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

    return bucket_1, bucket_2, bucket_3, bucket_0


def print_percentages(
    likely, very_likely, extremely_likely, none, contract_df, return_result=False
):
    not_likely_percentage = round(len(none) / len(contract_df) * 100, 2)
    likely_percentage = round(len(likely) / len(contract_df) * 100, 2)
    very_likely_percentage = round(len(very_likely) / len(contract_df) * 100, 2)
    extremely_likely_percentage = round(
        len(extremely_likely) / len(contract_df) * 100, 2
    )

    print("Not Likely: ", not_likely_percentage, "%")
    print("Could Contain: ", likely_percentage, "%")
    print("Likely: ", very_likely_percentage, "%")
    print("Very Likely: ", extremely_likely_percentage, "%")

    if return_result:
        return {
            "not_likely": not_likely_percentage,
            "likely": likely_percentage,
            "very_likely": very_likely_percentage,
            "extremely_likely": extremely_likely_percentage,
        }


def list_all_txt_files(base_dir):
    txt_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".txt"):
                relative_path = os.path.relpath(os.path.join(root, file), base_dir)
                txt_files.append(relative_path)
    return txt_files


def make_folders(likely, very_likely, extremely_likely, none, temp_dir, output_folder):
    folders = {
        "likely": os.path.join(output_folder, "likely"),
        "very_likely": os.path.join(output_folder, "very_likely"),
        "extremely_likely": os.path.join(output_folder, "extremely_likely"),
        "none": os.path.join(output_folder, "none"),
    }

    for folder in folders.values():
        os.makedirs(folder, exist_ok=True)

    # Create a mapping of file names to their relative paths
    uploaded_files = {
        os.path.basename(file): file for file in list_all_txt_files(temp_dir)
    }

    categories = {
        "likely": likely,
        "very_likely": very_likely,
        "extremely_likely": extremely_likely,
        "none": none,
    }

    for category, contracts in categories.items():
        for _, contract in contracts.iterrows():
            contract_id = contract["contract_ids"]
            if contract_id in uploaded_files:
                source_path = os.path.join(temp_dir, uploaded_files[contract_id])
                destination_path = os.path.join(folders[category], contract_id)
                shutil.copy(source_path, destination_path)
            else:
                print(f"File not found: {contract_id}")

    return (
        folders["likely"],
        folders["very_likely"],
        folders["extremely_likely"],
        folders["none"],
    )


def print_single(likely, very_likely, extremely_likely, none, return_result=False):
    result = ""
    if len(extremely_likely) != 0:
        result = "very likely"
        output = "very likely"
    elif len(very_likely) != 0:
        result = "likely"
        output = "likely"
    elif len(likely) != 0:
        result = "could contain"
        output = "could contain"
    elif len(none) != 0:
        result = "unlikely"
        output = "unlikely"
    else:
        result = "unknown"
        output = "unknown"
    if return_result:
        return result
    else:
        print(output)


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