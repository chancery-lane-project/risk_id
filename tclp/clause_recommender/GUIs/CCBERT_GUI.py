# legalBERTGUI.py

"""This is a file to launch a localGUI of the CCBERT approach.

Most users will be more interested in using the dockerized version but this is included for completeness."""

import os
import threading
import tkinter as tk
from tkinter import filedialog, ttk

import nltk
import numpy as np
from transformers import AutoModel, AutoTokenizer

from tclp.clause_recommender import utils

light_blue_text = "#e6f5ff"
mid_blue_text = "#b8e2ff"
navy = "#001f3f"
font = "Helvetica"


class TCLPApp:
    def __init__(self, root):
        self.root = root
        self.root.title("TCLP Clause Matcher")
        self.root.geometry("700x600")
        self.root.configure(bg=navy)

        # Configure style
        self.style = ttk.Style(self.root)
        self.style.theme_use("clam")
        self.style.configure(
            "TButton",
            font=(font, 12),
            padding=6,
            background=light_blue_text,
            foreground=navy,
            relief="flat",
            anchor="center",
        )
        self.style.configure(
            "TLabel", font=(font, 12), background=navy, foreground=light_blue_text
        )
        self.style.configure("TCombobox", font=(font, 12), padding=5)
        self.style.configure("TProgressbar", thickness=10)

        # Attributes for model, tokenizer, and data storage
        self.model = None
        self.tokenizer = None
        self.documents = []
        self.file_names = []
        self.embeddings = {}
        self.top_clause_text = ""
        self.contract_text = ""

        # Loading screen
        self.loading_screen = tk.Label(
            self.root,
            text="Preparing the application. This may take a few minutes the first time you open the app. After that, it will be instantaneous",
            font=(font, 12),
            wraplength=500,
            justify="center",
            bg=navy,
            fg=light_blue_text,
        )
        self.loading_screen.pack(pady=50)

        # Progress bar
        self.progress = ttk.Progressbar(
            self.root, orient="horizontal", mode="determinate", length=400
        )
        self.progress.pack(pady=20)

        # Start loading model in a separate thread
        self.load_model_thread = threading.Thread(target=self.load_model)
        self.load_model_thread.start()

    def load_model(self):
        local_model_dir = "../../CC_BERT/CC_model"
        embeddings_dir = "../../CC_BERT/CC_embeddings"
        self.progress["maximum"] = 6

        if not os.path.exists(local_model_dir):
            os.makedirs(local_model_dir)
            self.tokenizer = AutoTokenizer.from_pretrained(local_model_dir)
            self.model = AutoModel.from_pretrained(local_model_dir)
            self.tokenizer.save_pretrained(local_model_dir)
            self.model.save_pretrained(local_model_dir)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(local_model_dir)
            self.model = AutoModel.from_pretrained(local_model_dir)

        self.progress["value"] += 1
        self.root.update_idletasks()

        folder_path = "../../data/clause_boxes"
        self.documents, self.file_names = utils.load_clauses(folder_path)

        if not os.path.exists(embeddings_dir):
            os.makedirs(embeddings_dir)

        for method in ["cls", "mean", "max", "concat", "specific"]:
            embedding_path = os.path.join(embeddings_dir, f"{method}_embeddings.npy")
            if os.path.exists(embedding_path):
                self.embeddings[method] = np.load(embedding_path)
            else:
                self.embeddings[method] = np.vstack(
                    [
                        utils.encode_text(doc, self.tokenizer, self.model, method)
                        for doc in self.documents
                    ]
                )
                np.save(embedding_path, self.embeddings[method])

            self.progress["value"] += 1
            self.root.update_idletasks()

        self.loading_screen.pack_forget()
        self.progress.pack_forget()
        self.show_main_gui()

    def show_main_gui(self):
        self.banner = tk.Label(
            self.root,
            text="Remember these are just suggestions! Please review each clause for your specific needs.",
            font=(font, 14, "bold"),
            wraplength=600,
            fg="white",
            bg=navy,
        )
        self.banner.pack(pady=10)

        self.method_var = tk.StringVar()
        self.method_var.set("cls")
        method_label = ttk.Label(self.root, text="Select Embedding Method:")
        method_label.pack(pady=(10, 0))
        self.method_dropdown = ttk.Combobox(
            self.root,
            textvariable=self.method_var,
            values=["cls", "mean", "max", "concat", "specific"],
            state="readonly",
        )
        self.method_dropdown.pack(pady=10)

        self.browse_button = tk.Button(
            self.root,
            text="Browse File",
            command=self.browse_file,
            font=(font, 12),
            bg=light_blue_text,
            fg=navy,
            borderwidth=0,
            highlightthickness=0,
            relief="flat",
            padx=10,
            pady=5,
        )
        self.browse_button.pack(pady=10)

        self.view_contract_button = tk.Button(
            self.root,
            text="View Contract Text",
            command=self.view_contract,
            state=tk.DISABLED,
            font=(font, 12),
            bg=light_blue_text,
            fg=navy,
            borderwidth=0,
            highlightthickness=0,
            relief="flat",
            padx=10,
            pady=5,
        )
        self.view_contract_button.pack(pady=5)

        self.result_frame = tk.Frame(self.root, bg=navy)
        self.result_frame.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)

    def find_best_matching_clause(self, query, method):
        query_embedding = utils.encode_text(
            query, self.tokenizer, self.model, method
        ).reshape(1, -1)
        document_embeddings = self.embeddings[method]

        # Use get_matching_clause to find the best match
        _, _, _, similarities, _ = utils.get_matching_clause(
            query_embedding, document_embeddings, self.file_names
        )

        # Find top three matches
        best_match_names, best_match_scores, _ = utils.find_top_three(
            similarities, self.file_names
        )

        return best_match_names, best_match_scores

    def format_clause_name(self, filename):
        return filename.replace("_", " ").replace(".txt", "")

    def browse_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if file_path:
            with open(file_path, "r", encoding="utf-8") as file:
                query = file.read()
            self.contract_text = query
            self.view_contract_button.config(state=tk.NORMAL)
            selected_method = self.method_var.get()
            self.display_results(query, selected_method)

    def display_results(self, query, method):
        for widget in self.result_frame.winfo_children():
            widget.destroy()

        instruction_label = tk.Label(
            self.result_frame,
            text="Click on the name of the clauses below to see the full text of the top 3 matching clauses for your contract:",
            font=(font, 16, "bold"),
            fg=mid_blue_text,  # Slightly darker light blue
            bg=navy,
            wraplength=500,
        )
        instruction_label.pack(pady=5)

        best_match_names, best_match_scores = self.find_best_matching_clause(
            query, method
        )

        for i, (name, score) in enumerate(zip(best_match_names, best_match_scores), 1):
            display_name = name.replace(".txt", "")

            score_frame = tk.Frame(self.result_frame, bg=navy, padx=10, pady=5)
            score_frame.pack(pady=5, fill="x")

            score_label = tk.Label(
                score_frame,
                text=f"Similarity Score: {score:.4f}",
                font=(font, 12, "bold"),
                fg=light_blue_text,
                bg=navy,
            )
            score_label.pack(side="top", anchor="w")

            clause_button = tk.Button(
                score_frame,
                text=display_name,
                font=(font, 16),
                bg=light_blue_text,
                fg=navy,
                borderwidth=0,
                relief="flat",
                command=lambda n=name: self.view_clause_text(n),
            )
            clause_button.pack(side="bottom", anchor="w", fill="x", padx=5, pady=5)

    def view_contract(self):
        self.show_text_in_window("Contract Text", self.contract_text)

    def view_clause_text(self, clause_name):
        # Ensure clause_name is formatted to match entries in self.file_names
        formatted_name = clause_name.replace(" ", "_")
        if not formatted_name.endswith(".txt"):
            formatted_name += ".txt"
        try:
            clause_index = self.file_names.index(formatted_name)
            clause_text = self.documents[clause_index]
            self.show_text_in_window(f"Clause: {clause_name}", clause_text)
        except ValueError:
            print(f"Error: {formatted_name} not found in file names.")

    def show_text_in_window(self, title, content):
        new_window = tk.Toplevel(self.root)
        new_window.title(title)
        text_widget = tk.Text(
            new_window, wrap=tk.WORD, font=(font, 12), bg="white", fg="black"
        )
        text_widget.insert(tk.END, content)
        text_widget.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        text_widget.config(state=tk.DISABLED)


def main():
    nltk.download("punkt")
    root = tk.Tk()
    app = TCLPApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
