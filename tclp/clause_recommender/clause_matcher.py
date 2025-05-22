# clause_matcher.py

"""
This allows you to create an instance of the LegalBERTMatcher class and use it to find the best matching clause in a set of clauses for a given query text.

It is used for inserting clauses in the clause_detector class.

Most users will not have to use this file."""

import os
import numpy as np
from transformers import AutoTokenizer, AutoModel
import nltk
import utils

nltk.download("punkt")


class LegalBERTMatcher:
    def __init__(
        self,
        model_dir="../legalbert/legalbert_model",
        embeddings_dir="../legalbert/legalbert_embeddings",
    ):
        self.model_dir = model_dir
        self.embeddings_dir = embeddings_dir
        self.model = None
        self.tokenizer = None
        self.embeddings = {}
        self.file_names = []
        self.documents = []
        self.load_model()

    def load_model(self):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            self.tokenizer = AutoTokenizer.from_pretrained("casehold/legalbert")
            self.model = AutoModel.from_pretrained("casehold/legalbert")
            self.tokenizer.save_pretrained(self.model_dir)
            self.model.save_pretrained(self.model_dir)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            self.model = AutoModel.from_pretrained(self.model_dir)

    def load_clauses(self, folder_path):
        self.documents, self.file_names = utils.load_clauses(folder_path)
        for method in ["cls", "mean", "max", "concat", "specific"]:
            embedding_path = os.path.join(
                self.embeddings_dir, f"{method}_embeddings.npy"
            )
            if os.path.exists(embedding_path):
                self.embeddings[method] = np.load(embedding_path)
            else:
                self.embeddings[method] = np.vstack(
                    [
                        utils.encode_text(doc, self.tokenizer, self.model, method)
                        for doc in self.documents
                    ]
                )
                os.makedirs(self.embeddings_dir, exist_ok=True)
                np.save(embedding_path, self.embeddings[method])

    def find_best_matching_clause(self, query_text, method="cls"):
        query_embedding = utils.encode_text(
            query_text, self.tokenizer, self.model, method
        ).reshape(1, -1)
        document_embeddings = self.embeddings[method]
        _, _, _, similarities, _ = utils.get_matching_clause(
            query_embedding, document_embeddings, self.file_names
        )
        best_match_names, best_match_scores, _ = utils.find_top_three(
            similarities, self.file_names
        )
        return best_match_names, best_match_scores

    def match_clauses(self, contract_path, clause_folder_path, method="cls"):
        self.load_clauses(clause_folder_path)
        with open(contract_path, "r", encoding="utf-8") as file:
            contract_text = file.read()
        top_matches, top_scores = self.find_best_matching_clause(contract_text, method)
        formatted_matches = [
            (name.replace("_", " ").replace(".txt", ""), score)
            for name, score in zip(top_matches, top_scores)
        ]
        return formatted_matches
