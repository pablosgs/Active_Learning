from sentence_transformers import SentenceTransformer
import torch


class TextVectorization:
    def __init__(self) -> None:
        # Initialize SBERT model
        self.model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def vectorize(self, text):
        # Vectorize text with the SBERT model
        vectors = self.model.encode(text, convert_to_numpy=True, device=self.device)
        return vectors
