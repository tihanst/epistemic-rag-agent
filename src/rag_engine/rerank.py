from typing import List, Tuple

from sentence_transformers import CrossEncoder

from .docsource_connector import Corpus_Retrieval
from ..Config.config import Settings


class ReRanker:
    """
    ReRanker class for re-ranking

    """

    def __init__(self, settings: Settings):
        """
        Initialize ReRanker class
        Args:
            settings (Settings): Settings object
        Returns:
            None
        """
        self.model = CrossEncoder(settings.CROSS_ENCODER_MODEL)
        self.top_k: int = settings.CROSS_ENCODER_TOP_K

    def rerank(
        self,
        query: str,
        rrf_ranks: List[int],
        retrieved_corpus_info: List[Corpus_Retrieval],
        top_k: int,
    ) -> List[Tuple[int, str, Corpus_Retrieval]]:
        """
        Re-rank the documents using the CrossEncoder model.

        Args:
            query (str): The query string.
            rrf_ranks (List[int]): The list of document indices from reciprocal rank fusion process from best to worst.
            retrieved_corpus_info (List[Corpus_Retrieval]): The list of full data (chunk_id, ner_entities, norm_doc, compressed_doc, metadata) from the database for document access.
            top_k (Settings.CROSS_ENCODER_TOP_K): The number of documents to return, or 0 if whole list is to be returned.
        Returns:
            List[Tuple[int, str, Corpus_Retrieval]]: The list of tuples containing the document, the document id, and the full data from the database.
        """
        rerank_scores: List[float] = self.model.predict(
            [(query, retrieved_corpus_info[x]["norm_doc"]) for x in rrf_ranks]
        ).tolist()

        sorted_reranks = sorted(
            zip(rrf_ranks, rerank_scores), key=lambda x: x[1], reverse=True
        )

        return_reranks = sorted_reranks if top_k == 0 else sorted_reranks[:top_k]

        return [
            (x[0], retrieved_corpus_info[x[0]]["norm_doc"], retrieved_corpus_info[x[0]])
            for x in return_reranks
        ]
