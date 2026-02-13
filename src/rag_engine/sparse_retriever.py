# type: ignore
from typing import List, Union, Tuple, Sequence

import bm25s
import Stemmer

from ..Config.config import Settings

Corp = Sequence[Union[str, List[str]]]


class SparseRetriever:
    """
    Sparse retriever using BM25
    """

    def __init__(self, corpus: Corp, config: Settings):
        """
        Initialize the sparse retriever.

        Args:
            corpus (Corpus): The corpus to index.
            config (Settings): The configuration object.
        """

        self.corpus = corpus
        self.config = config
        self.stemmer: Stemmer.Stemmer = Stemmer.Stemmer("english")
        self.bm25s = bm25s
        self.bm25_retriever = bm25s.BM25()
        if isinstance(self.corpus[0], str):
            print("sparse for full string corpus")
            self.corpus_tokens = self.bm25s.tokenize(
                self.corpus, stopwords="en", stemmer=self.stemmer
            )
        else:
            assert isinstance(self.corpus[0], list), (
                "Corpus must be List[str] or List[List[str]]"
            )
            print("sparse for ner entities")
            self.corpus_tokens = self.bm25s.tokenize(
                [", ".join(x) for x in self.corpus],
                stopwords="en",
                stemmer=self.stemmer,
            )
        self.bm25_retriever.index(self.corpus_tokens)

    def bm25_retrieve(self, query: str, top_k: int) -> Tuple[List[int], List[float]]:
        """
        Retrieve the top k documents from the corpus that are most similar to the query based on BM25 scores.

        Args:
            query (str): The query to search for.
            top_k (int, optional): The number of documents to retrieve. Defaults to 100.

        Returns:
            A tuple of two lists, the first list contains the indices of the retrieved documents, the second list contains the BM25 scores of the retrieved documents.
        """

        query_tokens = self.bm25s.tokenize(
            [query], stopwords="en", stemmer=self.stemmer
        )
        indices, scores = self.bm25_retriever.retrieve(query_tokens, k=top_k)

        return indices.tolist()[0], scores.tolist()[0]
