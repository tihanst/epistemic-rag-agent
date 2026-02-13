from typing import List, Any
from numpy.typing import NDArray
import numpy as np


from langchain_core.embeddings import Embeddings
from FlagEmbedding import BGEM3FlagModel


class BGEM3F_Embed(Embeddings):
    """
    Class for embedding text using the BGEM3F model
    """

    def __init__(self):
        super().__init__()
        self.model: Any = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds a list of documents using the BGEM3F model

        Args:
            texts (List[str]): List of documents to embed

        Returns:
            List[List[float]]: List of embeddings
        """
        vecs: NDArray[np.float16] = self.model.encode(texts)["dense_vecs"]
        vecs2 = [x.tolist() for x in vecs]
        return vecs2

    def embed_query(self, text: str) -> List[float]:
        """
        Embeds a query using the BGEM3F model

        Args:
            text (str): Query to embed

        Returns:
            List[float]: Embedding
        """
        vec = self.model.encode(text)["dense_vecs"].tolist()
        return vec
