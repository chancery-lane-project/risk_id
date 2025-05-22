# doc2vec with GUI

"""This is a file to launch a local GUI of the doc2vec approach.

This was not the selected approach for the final output but is included for completeness."""

# necessary imports
import os
import tkinter as tk
from tkinter import filedialog, ttk

import nltk
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity


# Load the pre-trained Doc2Vec model (or train one if not available)
def load_model():
    folder_path = "/Users/georgia/Documents/Clause-Comparison/tclp/data/clause_boxes"
    documents = []
    file_names = []

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and filename.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as file:
                documents.append(file.read())
                file_names.append(filename)

    tagged_clauses = [
        TaggedDocument(words=word_tokenize(clause.lower()), tags=[str(i)])
        for i, clause in enumerate(documents)
    ]

    model = Doc2Vec(vector_size=50, min_count=1, epochs=20)
    model.build_vocab(tagged_clauses)
    model.train(tagged_clauses, total_examples=model.corpus_count, epochs=model.epochs)

    document_vectors = [model.dv[str(i)] for i in range(len(documents))]

    return model, document_vectors, documents, file_names


# Function to find the best matching documents
def find_best_matching_document(query, model, document_vectors, documents, file_names):
    query_vector = model.infer_vector(word_tokenize(query.lower()))
    query_vector = query_vector.reshape(1, -1)
    similarities = cosine_similarity(query_vector, document_vectors).flatten()

    best_match_indices = np.argsort(similarities)[::-1][:3]
    best_match_names = [file_names[i] for i in best_match_indices]
    best_match_scores = [similarities[i] for i in best_match_indices]
    best_match_texts = [documents[i] for i in best_match_indices]

    return best_match_names, best_match_scores, best_match_texts


# GUI Application
class TCLPApp:
    def __init__(self, root):
        self.root = root
        self.root.title("TCLP Clause Matcher")
        self.root.geometry("600x500")

        # Model and documents
        (
            self.model,
            self.document_vectors,
            self.documents,
            self.file_names,
        ) = load_model()
        self.top_clause_text = ""
        self.contract_text = ""

        # UI Elements
        self.label = ttk.Label(
            self.root,
            text="Drag and drop a file or click to browse",
            font=("Arial", 14),
        )
        self.label.pack(pady=20)

        self.browse_button = ttk.Button(
            self.root, text="Browse File", command=self.browse_file
        )
        self.browse_button.pack(pady=10)

        self.result_text = tk.Text(
            self.root, wrap=tk.WORD, height=10, font=("Arial", 12)
        )
        self.result_text.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)

        # Buttons to view texts
        self.view_clause_button = ttk.Button(
            self.root,
            text="Click to view top clause text",
            command=self.view_top_clause,
            state=tk.DISABLED,
        )
        self.view_clause_button.pack(pady=5)

        self.view_contract_button = ttk.Button(
            self.root,
            text="Click to view contract text",
            command=self.view_contract,
            state=tk.DISABLED,
        )
        self.view_contract_button.pack(pady=5)

    def browse_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if file_path:
            with open(file_path, "r", encoding="utf-8") as file:
                query = file.read()
            self.contract_text = query
            self.display_results(query)

    def display_results(self, query):
        (
            best_match_names,
            best_match_scores,
            best_match_texts,
        ) = find_best_matching_document(
            query,
            self.model,
            self.document_vectors,
            self.documents,
            self.file_names,
        )
        self.result_text.delete(1.0, tk.END)

        # Check if any score is above 0.5
        if all(score <= 0.5 for score in best_match_scores):
            self.result_text.insert(
                tk.END,
                "Sorry! It looks like there aren't any good matches for your contract.\n",
            )
            # Disable buttons if no good match
            self.view_clause_button.config(state=tk.DISABLED)
            self.view_contract_button.config(state=tk.DISABLED)
            return

        self.top_clause_text = best_match_texts[0]

        self.result_text.insert(
            tk.END, "Top 3 matching clauses and their similarity scores:\n\n"
        )
        for i, (name, score) in enumerate(zip(best_match_names, best_match_scores), 1):
            self.result_text.insert(tk.END, f"{i}. Clause: {name}\n")
            self.result_text.insert(tk.END, f"   Similarity Score: {score:.4f}\n\n")

        # Enable buttons
        self.view_clause_button.config(state=tk.NORMAL)
        self.view_contract_button.config(state=tk.NORMAL)

    def view_top_clause(self):
        self.show_text_in_window("Top Clause Text", self.top_clause_text)

    def view_contract(self):
        self.show_text_in_window("Contract Text", self.contract_text)

    def show_text_in_window(self, title, content):
        new_window = tk.Toplevel(self.root)
        new_window.title(title)
        text_widget = tk.Text(new_window, wrap=tk.WORD, font=("Arial", 12))
        text_widget.insert(tk.END, content)
        text_widget.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        text_widget.config(state=tk.DISABLED)


# Main function to run the GUI
def main():
    nltk.download("punkt")  # Ensure nltk is ready
    root = tk.Tk()
    app = TCLPApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
