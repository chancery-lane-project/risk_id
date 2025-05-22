import os
import pickle

import nltk
import numpy as np
import utils
from sklearn.preprocessing import normalize
from transformers import AutoModel, AutoTokenizer

nltk.download("punkt")


class CCMatcher:
    def __init__(
        self,
        model_dir="../../CC_BERT/CC_model",
        embeddings_dir="../../CC_BERT/CC_embeddings",
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
            raise FileNotFoundError(f"Model directory {self.model_dir} does not exist.")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModel.from_pretrained(self.model_dir)

    def load_clauses(self, folder_path, methods=["cls"]):
        meta_path = os.path.join(self.embeddings_dir, "clause_texts.pkl")
        if os.path.exists(meta_path):
            with open(meta_path, "rb") as f:
                data = pickle.load(f)
            self.file_names = data["file_names"]
            self.documents = data["texts"]
        else:
            self.documents, self.file_names = utils.load_clauses(folder_path)
            os.makedirs(self.embeddings_dir, exist_ok=True)
            with open(meta_path, "wb") as f:
                pickle.dump({"file_names": self.file_names, "texts": self.documents}, f)

        for method in methods:
            embedding_path = os.path.join(self.embeddings_dir, f"{method}_embeddings.npy")
            if os.path.exists(embedding_path):
                self.embeddings[method] = np.load(embedding_path)
            else:
                self.embeddings[method] = np.vstack(
                    [utils.encode_text(doc, self.tokenizer, self.model, method) for doc in self.documents]
                )
                np.save(embedding_path, self.embeddings[method])

    def find_best_matching_clause(self, query_text, method="cls"):
        query_embedding = utils.encode_text(
            query_text, self.tokenizer, self.model, method
        ).reshape(1, -1)
        query_embedding = normalize(query_embedding)
        document_embeddings = normalize(self.embeddings[method])
        _, _, _, similarities, _ = utils.get_matching_clause(
            query_embedding, document_embeddings, self.file_names
        )
        best_match_names, best_match_scores, _ = utils.find_top_three(
            similarities, self.file_names
        )
        return best_match_names, best_match_scores

    def match_clauses(self, contract_path, clause_folder_path, method="cls"):
        self.load_clauses(clause_folder_path, methods=[method])
        with open(contract_path, "r", encoding="utf-8") as file:
            contract_text = file.read()
        top_matches, top_scores = self.find_best_matching_clause(contract_text, method)
        formatted_matches = [
            (name.replace("_", " ").replace(".txt", ""), score)
            for name, score in zip(top_matches, top_scores)
        ]
        return formatted_matches
