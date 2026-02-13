from __future__ import annotations
from typing import List, Dict, Tuple, Union, TypedDict


from ..Config.config import Settings
from .docsource_connector import Source
from .embedder import BGEM3F_Embed
from .sparse_retriever import SparseRetriever
from . import extraction_formatter as ef
from .rerank import ReRanker
from .fuse import RRF
from ..LLM.promp_templates import PROMPT_START


Idx2 = Tuple[int, int]
Idx3 = Tuple[int, int, int]
Span = Union[Idx2, Idx3]
RAGOutput = Tuple[str, List[Tuple[Span, str]]]


class Data(TypedDict):
    chunk_id: int
    ner_entities: List[str]
    norm_doc: str
    compressed_doc: str
    metadata: Dict[str, int]


class RAGPipeline:
    """
    RAGPipeline class for RAG pipeline
    """

    def __init__(self):
        """
        Initialize RAGPipeline class
        """
        self.settings = Settings()
        self.embedder = BGEM3F_Embed()
        self.source = Source(self.settings, self.embedder)
        self.data = [
            Data(x) for x in self.source.corpus_retrieval()
        ]  # Will have to make sure database is being served
        self.full_corpus = [x["norm_doc"] for x in self.data]
        self.ner_corpus = [x["ner_entities"] for x in self.data]
        self.full_sparse_retriever = SparseRetriever(self.full_corpus, self.settings)
        self.ner_sparse_retriever = SparseRetriever(self.ner_corpus, self.settings)
        self.reranker = ReRanker(self.settings)
        self.fuser = RRF()

    def execute_extraction(self, query: str) -> RAGOutput:
        """
        Execute RAG pipeline extraction.

        Args:
            query (str): The query to be processed.
        Returns:
            RAGOutput: A tuple containing the query and a list of tuples, where each tuple contains a span and a string.

        """
        raw_dense_retrieve: List[Tuple[int, float, str, Dict[str, int]]] = (
            self.source.dense_retrieval(query, "raw")
        )
        compressed_dense_retrieve: List[Tuple[int, float, str, Dict[str, int]]] = (
            self.source.dense_retrieval(query, "compressed")
        )

        retrieved_full_sparse: Tuple[List[int], List[float]] = (
            self.full_sparse_retriever.bm25_retrieve(query, top_k=100)
        )
        retrieved_ner_sparse: Tuple[List[int], List[float]] = (
            self.ner_sparse_retriever.bm25_retrieve(query, top_k=100)
        )

        raw_dense_list = [x[-1]["doc_idx"] for x in raw_dense_retrieve]
        # Bug was in here [x[0] for x in raw_dense_retrieve]
        comp_dense_list = [x[-1]["doc_idx"] for x in compressed_dense_retrieve]
        # Bug was in here [x[0] for x in compressed_dense_retrieve]
        full_corpus_sparse_list = retrieved_full_sparse[0]
        ner_entities_sparse_list = retrieved_ner_sparse[0]

        fused_list: Tuple[List[Tuple[int, float]], List[int]] = self.fuser.rrf_fuse(
            [
                raw_dense_list,
                comp_dense_list,
                full_corpus_sparse_list,
                ner_entities_sparse_list,
            ]
        )

        final: List[Tuple[int, str, Data]] = self.reranker.rerank(
            query, fused_list[1], self.data, top_k=25
        )

        sfinal: List[Tuple[int, str, Data]] = sorted(final, key=lambda x: x[0])

        expanded_merged = ef.expand_and_merge_linear(sfinal, self.full_corpus)

        reduced = ef.reduce_and_return(expanded_merged)

        final = ef.clean_gap_one_merges(reduced)

        return (query, final)

    # Unused but keep for reference
    @staticmethod
    def format_context_for_llm(
        query_and_context: Tuple[
            str, List[Tuple[Tuple[int, ...], str]]
        ],  # Tuple[str, List[Tuple[Span, str]]]
    ) -> str:
        """
        Format context for LLM
        Args:
            query_and_context (Tuple[str, List[Tuple[Span, str]]]): A tuple containing the query and a list of tuples, where each tuple contains a span and a string.
        Returns:
            str: A string containing the formatted context.
        """
        big_prompt = (
            PROMPT_START
            + "\n\n"
            + "Question: "
            + query_and_context[0]
            + "\n\n\n"
            + "Excerpts:"
            + "\n\n"
            + "```"
            + "\n\n"
        )

        for x in enumerate(query_and_context[1]):
            if x[0] < len(query_and_context[1]) - 1:
                big_prompt += (
                    f"#Excerpt {x[0] + 1:}#"
                    + "\n\n"
                    + x[1][1]
                    + "\n\n\n"
                    + "------------------"
                    + "\n\n"
                )
            else:
                big_prompt += f"#Excerpt {x[0] + 1:}#\n\n" + x[1][1] + "\n\n" + "```"

        return big_prompt

    @staticmethod
    def format_context_for_llm_excerpts(
        query_and_context: Tuple[
            str, List[Tuple[Tuple[int, ...], str]]
        ],  # Tuple[str, List[Tuple[Span, str]]]
    ) -> str:
        """
        Format context for LLM
        Args:
            query_and_context (Tuple[str, List[Tuple[Span, str]]]): A tuple containing the query and a list of tuples, where each tuple contains a span and a string.
        Returns:
            str: A string containing the formatted context.
        """
        big_prompt = (
            PROMPT_START
            + "\n\n"
            + "Question: "
            + query_and_context[0]
            + "\n\n\n"
            + "Excerpts:"
            + "\n\n"
            + "```"
            + "\n\n"
        )

        for x in enumerate(query_and_context[1]):
            if x[0] < len(query_and_context[1]) - 1:
                big_prompt += (
                    f"#Excerpt {x[1][0]:}#"
                    + "\n\n"
                    + x[1][1]
                    + "\n\n\n"
                    + "------------------"
                    + "\n\n"
                )
            else:
                big_prompt += f"#Excerpt {x[1][0]}#\n\n" + x[1][1] + "\n\n" + "```"

        return big_prompt
