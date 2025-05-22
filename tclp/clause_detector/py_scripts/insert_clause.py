# insert_clause.py
"""This script is designed to utilize the LegalBERTMatcher class to insert clauses into contracts.

You do not need to run this unless you want to create your own synthetic dataset.

The created synthetic dataset will be available in the online data storage if you just want to download it."""

import os
import random
import argparse
from tclp.clause_recommender.clause_matcher import LegalBERTMatcher

matcher = LegalBERTMatcher()
clauses_folder_path = "../../data/cleaned_clauses_detect"
generated_clause_folder_path = "../../data/cleaned_gen_clauses"
modified_folder_path = "../../data/synth_data/modified_real"
untouched_folder_path = "../../data/synth_data/untouched"
modified_gen_folder_path = "../../data/synth_data/modified_gen"
combined_folder_path = "../../data/synth_data/combined"

os.makedirs(modified_folder_path, exist_ok=True)
os.makedirs(untouched_folder_path, exist_ok=True)
os.makedirs(modified_gen_folder_path, exist_ok=True)
os.makedirs(combined_folder_path, exist_ok=True)


def get_unique_filename(output_folder, filename):
    """Generate a unique filename by appending a suffix if the file already exists."""
    base_name, extension = os.path.splitext(filename)
    counter = 1
    unique_filename = filename

    while os.path.exists(os.path.join(output_folder, unique_filename)):
        unique_filename = f"{base_name}_{counter}{extension}"
        counter += 1

    return unique_filename


def insert_randomly(contract_path, clauses, clause_folder_path):
    with open(contract_path, "r", encoding="utf-8") as contract_file:
        text = contract_file.read()

    clause_count = 2 if random.random() < 0.2 else 1  # 20% chance for two clauses
    to_insert = random.sample(clauses, clause_count)

    clause_contents = []
    for clause_file in to_insert:
        clause_path = os.path.join(clause_folder_path, clause_file)
        with open(clause_path, "r", encoding="utf-8") as clause_file:
            clause_contents.append(clause_file.read())

    lines = text.splitlines()
    paragraph_breaks = [
        i
        for i in range(1, len(lines) - 1)
        if lines[i].strip() == "" and lines[i - 1].strip() and lines[i + 1].strip()
    ]
    index = random.choice(paragraph_breaks) if paragraph_breaks else len(lines)

    labeled_text = [(line, "0") for line in lines[:index]]
    labeled_text += [("", "0")]
    for clause_content in clause_contents:
        clause_lines = clause_content.splitlines()
        labeled_text += [(line, "1") for line in clause_lines]
        labeled_text += [("", "0")]
    labeled_text += [(line, "0") for line in lines[index:]]

    return labeled_text, clause_count


def label_0s(contract_path):
    with open(contract_path, "r", encoding="utf-8") as contract_file:
        text = contract_file.read()
    lines = text.splitlines()
    labeled_text = [(line, "0") for line in lines]
    return labeled_text


def process_contracts_folder(
    contracts_folder_path,
    clauses_folder_path,
    generated_clause_folder_path,
    modified_folder,
    modified_gen_folder,
    untouched_folder,
    combined_folder,
):
    os.makedirs(modified_folder, exist_ok=True)
    os.makedirs(untouched_folder, exist_ok=True)
    os.makedirs(modified_gen_folder, exist_ok=True)
    os.makedirs(combined_folder, exist_ok=True)

    # Load clause filenames
    real_clauses = os.listdir(clauses_folder_path)
    gen_clauses = os.listdir(generated_clause_folder_path)

    for root, _, files in os.walk(contracts_folder_path):
        for contract_filename in files:
            if not contract_filename.lower().endswith(".txt"):
                print(f"Skipping non-text file: {contract_filename}")
                continue

            contract_path = os.path.join(root, contract_filename)
            if not os.path.isfile(contract_path):
                continue

            # Determine the modification type based on probabilities
            r = random.random()
            clause_count = 0
            modify_real = False
            modify_gen = False

            if r < 0.4:
                # Leave untouched
                labeled_content = label_0s(contract_path)
                output_folder = untouched_folder
            elif r < 0.7:
                # Modify with real clauses
                modify_real = True
                try:
                    top_clauses = matcher.match_clauses(
                        contract_path, clauses_folder_path, method="cls"
                    )
                    top_clauses = [
                        clause[0].replace(" ", "_") + ".txt" for clause in top_clauses
                    ]
                    top_clauses = [
                        clause for clause in top_clauses if clause in real_clauses
                    ]

                    if not top_clauses:
                        print(
                            f"No matched clauses found for {contract_filename}. Skipping."
                        )
                        continue

                    labeled_content, clause_count = insert_randomly(
                        contract_path, top_clauses, clauses_folder_path
                    )
                    output_folder = modified_folder
                except Exception as e:
                    print(f"Error processing real clauses for {contract_filename}: {e}")
                    continue
            else:
                # Modify with generated clauses
                modify_gen = True
                try:
                    labeled_content, clause_count = insert_randomly(
                        contract_path, gen_clauses, generated_clause_folder_path
                    )
                    output_folder = modified_gen_folder
                except Exception as e:
                    print(
                        f"Error processing generated clauses for {contract_filename}: {e}"
                    )
                    continue

            # Generate unique filename in the combined folder
            unique_filename = get_unique_filename(combined_folder, contract_filename)
            combined_contract_path = os.path.join(combined_folder, unique_filename)

            # Save the file in the combined folder first
            with open(combined_contract_path, "w", encoding="utf-8") as combined_file:
                combined_file.write(
                    "\n".join(f"{label} {line}" for line, label in labeled_content)
                )

            # Save the file in the specific folder with the same name
            output_contract_path = os.path.join(output_folder, unique_filename)
            with open(output_contract_path, "w", encoding="utf-8") as specific_file:
                specific_file.write(
                    "\n".join(f"{label} {line}" for line, label in labeled_content)
                )

            # Logging
            if modify_real or modify_gen:
                clause_type = "gen" if modify_gen else "real"
                clause_msg = (
                    f"{clause_count} clause(s)" if clause_count > 1 else "1 clause"
                )
                print(f"Modified {contract_filename}: {clause_msg} ({clause_type}).")
            else:
                print(f"Left {contract_filename} untouched.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process a folder of contracts, randomly modifying them with clauses."
    )
    parser.add_argument(
        "contracts_folder",
        nargs="?",
        help="Path to the folder containing contract files",
    )
    args = parser.parse_args()

    if not args.contracts_folder:
        args.contracts_folder = input(
            "Please provide a contracts folder path: "
        ).strip()

    if not args.contracts_folder:
        print("Error: Contracts folder path is required.")
    else:
        process_contracts_folder(
            args.contracts_folder,
            clauses_folder_path,
            generated_clause_folder_path,
            modified_folder_path,
            modified_gen_folder_path,
            untouched_folder_path,
            combined_folder_path,
        )
