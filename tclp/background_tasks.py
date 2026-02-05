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

        response = {
            "classification": result,
            "highlighted_output_url": f"output/{output_filename}",
            "bucket_labels": {
                "cat0": CAT0,
                "cat1": CAT1,
                "cat2": CAT2,
                "cat3": CAT3
            },
            "file_hash": file_hash  # Include hash for find_clauses to use
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
