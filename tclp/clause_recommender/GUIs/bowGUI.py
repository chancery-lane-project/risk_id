# bag of words with GUI
"""This is a file to launch a local GUI of the bag of words approach.

This was not the selected approach for the final output but is included for completeness."""

import tkinter as tk
from tkinter import filedialog, ttk

from tclp.clause_recommender import utils


# GUI Application
class SimilarityApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Document Similarity Checker")
        self.root.geometry("600x700")

        self.documents, self.file_names = utils.load_clauses(
            "/Users/georgia/Documents/Clause-Comparison/tclp/data/clause_boxes"
        )
        self.merged_df = None
        self.top_clause_text = ""
        self.contract_text = ""

        # UI Elements
        self.label = ttk.Label(
            self.root, text="Select a document to compare", font=("Arial", 14)
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

        self.view_chart_button = ttk.Button(
            self.root,
            text="View Feature Chart",
            command=self.view_feature_chart,
            state=tk.DISABLED,
        )
        self.view_chart_button.pack(pady=5)

        self.view_clause_button = ttk.Button(
            self.root,
            text="View Top Clause",
            command=self.view_top_clause,
            state=tk.DISABLED,
        )
        self.view_clause_button.pack(pady=5)

        self.view_contract_button = ttk.Button(
            self.root,
            text="View Uploaded Document",
            command=self.view_contract,
            state=tk.DISABLED,
        )
        self.view_contract_button.pack(pady=5)

    def browse_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if file_path:
            with open(file_path, "r", encoding="utf-8") as file:
                self.contract_text = file.read()
            self.display_results(self.contract_text)

    def display_results(self, target_doc):
        top_matches = utils.find_top_similar_bow(
            target_doc, self.documents, self.file_names
        )

        # Extract elements from the returned dictionary
        top_names = top_matches["Top_Matches"]
        top_scores = top_matches["Scores"]
        top_texts = top_matches["Documents"]
        feature_chart = top_matches["Feature_Chart"]

        self.result_text.delete(1.0, tk.END)

        if top_names:
            self.top_clause_text = top_texts[0]
            self.result_text.insert(
                tk.END, "Top 3 matching clauses and their similarity scores:\n\n"
            )
            for i, (name, score) in enumerate(zip(top_names, top_scores), 1):
                self.result_text.insert(tk.END, f"{i}. Clause: {name}\n")
                self.result_text.insert(tk.END, f"   Similarity Score: {score:.4f}\n\n")

            self.merged_df = feature_chart
            self.view_chart_button.config(state=tk.NORMAL)
            self.view_clause_button.config(state=tk.NORMAL)
            self.view_contract_button.config(state=tk.NORMAL)
        else:
            self.result_text.insert(
                tk.END,
                "Sorry! It looks like there aren't any good matches for your contract.\n",
            )
            self.view_chart_button.config(state=tk.DISABLED)
            self.view_clause_button.config(state=tk.DISABLED)
            self.view_contract_button.config(state=tk.DISABLED)

    def view_feature_chart(self):
        if self.merged_df is not None:
            new_window = tk.Toplevel(self.root)
            new_window.title("Feature Chart")
            text_widget = tk.Text(new_window, wrap=tk.WORD, font=("Arial", 12))
            text_widget.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

            text_widget.insert(
                tk.END, "Top overlapping words contributing to similarity:\n\n"
            )
            for _, row in self.merged_df.head(10).iterrows():
                text_widget.insert(
                    tk.END,
                    f"{row['word']}: {row['target_frequency']} (target), {row['similar_frequency']} (similar)\n",
                )
            text_widget.config(state=tk.DISABLED)

    def view_top_clause(self):
        self.show_text_in_window("Top Clause Text", self.top_clause_text)

    def view_contract(self):
        self.show_text_in_window("Uploaded Document Text", self.contract_text)

    def show_text_in_window(self, title, content):
        new_window = tk.Toplevel(self.root)
        new_window.title(title)
        text_widget = tk.Text(new_window, wrap=tk.WORD, font=("Arial", 12))
        text_widget.insert(tk.END, content)
        text_widget.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        text_widget.config(state=tk.DISABLED)


# Main function to run the GUI
def main():
    root = tk.Tk()
    app = SimilarityApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
