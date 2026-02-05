"""
Background task processing for long-running ML operations.
"""

import hashlib
import os
import shutil
import time
import traceback
from typing import Dict, Any

import task_manager


def process_contract_task(
    task_id: str,
    file_content: bytes,
    filename: str,
    temp_dir: str,
    output_dir: str,
    tokenizer, d_model, c_model,
    embedding_cache: Dict,
    CAT0: str, CAT1: str, CAT2: str, CAT3: str
):
    """
    Background task to process a contract file.

    This function runs in a background thread and updates task status in the database.
    """
    import utils

    try:
        # Capture start time for analytics
        start_time = time.time()

        # Update status to processing
        task_manager.update_task(task_id, "processing", progress=10)

        # Create task-specific temp directory to avoid conflicts
        task_temp_dir = os.path.join(temp_dir, task_id)
        os.makedirs(task_temp_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        # Save original file first (for MarkItDown to process)
        original_file_path = os.path.join(task_temp_dir, filename)
        with open(original_file_path, "wb") as f:
            f.write(file_content)

        task_manager.update_task(task_id, "processing", progress=20)

        # Convert file to text for processing
        try:
            full_contract_text = utils.convert_file_to_text(file_content, filename)
        except Exception as e:
            raise Exception(f"Error converting file to text: {str(e)}")

        # Save as .txt file for load_unlabelled_contract to process
        base_filename = os.path.splitext(filename)[0]
        txt_filename = f"{base_filename}.txt"
        txt_file_path = os.path.join(task_temp_dir, txt_filename)
        with open(txt_file_path, "w", encoding="utf-8") as f:
            f.write(full_contract_text)

        task_manager.update_task(task_id, "processing", progress=30)

        # Convert to markdown using MarkItDown for display
        markdown_content = None
        try:
            from markitdown import MarkItDown

            md = MarkItDown()
            result = md.convert(original_file_path)

            if result and hasattr(result, 'text_content'):
                markdown_content = result.text_content
            elif result and hasattr(result, 'markdown'):
                markdown_content = result.markdown
            elif result:
                markdown_content = str(result)
                if markdown_content == "None" or not markdown_content.strip():
                    markdown_content = None

            if not markdown_content or str(markdown_content).strip() == "":
                markdown_content = None
        except Exception as e:
            print(f"Warning: MarkItDown conversion failed: {str(e)}")
            markdown_content = None

        task_manager.update_task(task_id, "processing", progress=40)

        # Embed the full contract text once
        file_hash = hashlib.md5(file_content).hexdigest()
        full_contract_embedding = utils.encode_text(full_contract_text, tokenizer, c_model)
        embedding_cache[file_hash] = {
            'embedding': full_contract_embedding,
            'text': full_contract_text
        }

        task_manager.update_task(task_id, "processing", progress=50)

        processed_contracts = utils.load_unlabelled_contract(task_temp_dir)
        texts = processed_contracts["text"].tolist()

        # Model predictions
        import torch
        device = torch.device("cpu")
        results, _ = utils.predict_climatebert(texts, tokenizer, device, d_model)
        result_df, _ = utils.create_result_df(results, processed_contracts)

        task_manager.update_task(task_id, "processing", progress=70)

        # Extract sentences that should be highlighted
        highlighted_sentences = result_df[
            (result_df['prediction'] == 1) | result_df['contains_climate_keyword']
        ]['sentence'].tolist()

        # Render document
        if markdown_content and str(markdown_content).strip():
            try:
                highlighted_output = utils.render_markdown_with_highlights(
                    markdown_content,
                    full_contract_text,
                    highlighted_sentences
                )
            except Exception as e:
                print(f"Warning: MarkItDown rendering failed: {str(e)}. Using fallback method.")
                highlighted_output = utils.highlight_climate_content(result_df)
        else:
            highlighted_output = utils.highlight_climate_content(result_df)

        task_manager.update_task(task_id, "processing", progress=85)

        # Save into output directory with timestamp
        timestamp = int(time.time())
        output_filename = f"highlighted_output_{timestamp}.html"
        filepath = os.path.join(output_dir, output_filename)
        utils.save_file(filepath, highlighted_output)
        print(f"[INFO] Saved highlighted output to: {filepath}")

        contract_df = utils.create_contract_df(
            result_df, processed_contracts, labelled=False
        )

        zero, one, two, three = utils.create_threshold_buckets(contract_df)

        result = utils.print_single(
            zero, one, two, three, return_result=True
        )

        # Calculate statistics
        word_count = len(full_contract_text.split())
        page_count = max(1, round(word_count / 250))  # Estimate ~250 words per page
        analysis_time = round(time.time() - start_time, 1)

        response = {
            "classification": result,
            "highlighted_output_url": f"output/{output_filename}",
            "bucket_labels": {
                "cat0": CAT0,
                "cat1": CAT1,
                "cat2": CAT2,
                "cat3": CAT3
            },
            "file_hash": file_hash,  # Include hash for find_clauses to use
            "statistics": {
                "word_count": word_count,
                "page_count": page_count,
                "analysis_time": analysis_time
            }
        }

        # Cleanup task temp directory
        shutil.rmtree(task_temp_dir, ignore_errors=True)

        # Mark task as completed with result
        task_manager.update_task(task_id, "completed", result=response, progress=100)
        print(f"[INFO] Task {task_id} completed successfully")

    except Exception as e:
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        print(f"[ERROR] Task {task_id} failed: {error_msg}")
        task_manager.update_task(task_id, "failed", error=str(e))


def find_clauses_task(
    task_id: str,
    file_content: bytes,
    filename: str,
    tokenizer, c_model, clause_tags, clf, umap_model,
    docs, names, name_to_child, name_to_url, emission_df,
    client, OPENROUTER_MODEL, DEFAULT_MODEL
):
    """
    Background task to find matching clauses for a contract.

    This function runs in a background thread and updates task status in the database.
    """
    import utils
    import pandas as pd
    import traceback

    try:
        # Update status to processing
        task_manager.update_task(task_id, "processing", progress=10)

        # Convert file to text
        try:
            query_text = utils.convert_file_to_text(file_content, filename)
        except Exception as e:
            raise Exception(f"Error converting file to text: {str(e)}")

        task_manager.update_task(task_id, "processing", progress=20)

        # Check for cached embedding
        file_hash = hashlib.md5(file_content).hexdigest()
        from app import embedding_cache  # Import here to avoid circular dependency

        if file_hash in embedding_cache:
            query_embedding = embedding_cache[file_hash]['embedding']
            del embedding_cache[file_hash]  # Clean up cache
        else:
            query_embedding = utils.encode_text(query_text, tokenizer, c_model)

        task_manager.update_task(task_id, "processing", progress=40)

        # Perform clustering
        subset_docs, subset_names, _ = utils.perform_cluster(
            clf, query_embedding, tokenizer, c_model, clause_tags, umap_model, embed=False
        )

        task_manager.update_task(task_id, "processing", progress=55)

        # Find similar documents
        bow_results = utils.find_top_similar_bow(
            target_doc=query_text, documents=docs, file_names=names,
            similarity_threshold=0.1, k=20
        )
        top_docs = bow_results["Documents"]
        top_names = bow_results["Top_Matches"]
        top_names_bow, _, top_texts_bow = utils.get_embedding_matches_subset(
            query_embedding, top_docs, top_names, tokenizer, c_model, k=5
        )

        task_manager.update_task(task_id, "processing", progress=70)

        # Combine results
        df_cluster = pd.DataFrame({
            "text": subset_docs,
            "source_name": subset_names,
            "matched_by": ["cluster"] * len(subset_names)
        })

        df_bow = pd.DataFrame({
            "text": top_texts_bow,
            "source_name": top_names_bow,
            "matched_by": ["bow"] * len(top_texts_bow)
        }).head(5)

        combined_df = pd.concat([df_cluster, df_bow], ignore_index=True)

        query_text_short = query_text[:1000]

        task_manager.update_task(task_id, "processing", progress=80)

        # Call LLM for recommendations
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
            model=OPENROUTER_MODEL,
            messages=messages,
            temperature=0,
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

        clause_block += '''Select the clauses from the list that best align with the contract.
    It is really important that you answer this consistently and the same way every time. If I upload the same contract against, I expect to see the same answer.

    This is a two step process.

    Step 1: Binary select the clauses that are a good fit for the contract. Go through one by one and remember which ones you selected as a potential fit. As a rule of thumb, give no fewer than 3 and no more than 7. If there is good reason, you can do fewer or more.
    Step 2: Go through those that you have selected as a fit and provide reasoning. Feel free to reconsider whether they are a fit once you go through them again.

    Before you being, read the rules below. They should guide you on both steps.

    Follow these rules:

    1. Your response must be a JSON of exactly as many objects as there are clauses you have selected as a fit, each with the keys "Clause Name" and "Reasoning".
    3. Only select from the clauses provided — do not invent new ones.
    4. Remember the contract's **content and purpose**. Their goal is likely not to reduce their emissions, but to meet other business or legal needs. We are telling them where they can inject climate-aligned language into the existing contract but the existing contract and its goals are the most important consideration.
    5. Pay close attention to what the contract is **doing** — the transaction type, structure, and key obligations — not just who the parties are or what sector they operate in.
    - Clauses must fit the **actual function and scope** of the contract.
    - For example, do not recommend a clause about land access if the contract is about software licensing.
    - Another example: do not recommend a clause about insurance if the contract is establishing a joint venture.
    6. Consider the relationship between the parties (e.g. supplier–customer, insurer–insured, JV partners).
    - If a clause assumes a different relationship, only suggest it if it can **realistically be adapted**, and explain how.
    7. You may include a clause that is not a perfect match if:
    - It serves a similar **legal or operational function**, and
    - You clearly explain how it could be adapted to the contract context.
    8. Do not recommend clauses that clearly mismatch the contract's type, scope, or parties.
    9. Avoid redundancy. If the contract already addresses a topic (e.g. dispute resolution), only suggest a clause on that topic if it adds clear value.

    Focus on legal function, contextual fit, and the actual mechanics of the contract. You are recommending **starting points** — plausible clauses the user could adapt.'''

        messages.append({"role": "user", "content": clause_block})

        task_manager.update_task(task_id, "processing", progress=90)

        response = client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=messages,
            temperature=0,
        )

        response_text = response.choices[0].message.content
        df_response = utils.parse_response(response_text)

        if df_response is None:
            print("Failed to parse LLM response, returning empty recommendations")
            result = {"matches": []}
            task_manager.update_task(task_id, "completed", result=result, progress=100)
            return

        # Check for missing clauses and retry if needed
        missing = []
        for clause in df_response["Clause Name"]:
            target = clause + ".txt"
            close = utils.get_close_matches(target, names, n=1, cutoff=0.8)
            if not close:
                missing.append(clause)

        if missing:
            print(f"[WARNING] Clauses not found: {missing}")
            messages.append({"role": "assistant", "content": response_text})
            messages.append({
                "role": "user",
                "content": (
                    "One of the clauses you recommended "
                    f"({', '.join(missing)}) was not in the provided set. "
                    "Do not hallucinate: only pick from the list I gave you, "
                    "and please try again."
                )
            })

            retry = client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=messages,
                temperature=0
            )
            response_text = retry.choices[0].message.content
            df_response = utils.parse_response(response_text)

            if df_response is None:
                print("Failed to parse retry LLM response, returning empty recommendations")
                result = {"matches": []}
                task_manager.update_task(task_id, "completed", result=result, progress=100)
                return

        # Get emission labels
        df_response = utils.get_emission_label(df_response, emission_df)

        # Build matches
        matches = []
        for _, row in df_response.iterrows():
            clause_name = row["Clause Name"].replace(".txt", "")
            matches.append({
                "name": clause_name,
                "child_name": name_to_child.get(clause_name, ""),
                "clause_url": name_to_url.get(clause_name, ""),
                "reason": row["Reasoning"],
                "emissions_sources": utils.parse_emissions_sources(row.get("combined_labels"))
            })

        print(f"[INFO] Returning {len(matches)} clause matches")
        print(f"[DEBUG] Matches data: {matches}")

        result = {"matches": matches}
        task_manager.update_task(task_id, "completed", result=result, progress=100)
        print(f"[INFO] Task {task_id} completed successfully")

    except Exception as e:
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        print(f"[ERROR] Task {task_id} failed: {error_msg}")
        task_manager.update_task(task_id, "failed", error=str(e))
