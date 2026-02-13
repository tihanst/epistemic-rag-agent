from typing import List, Literal, Tuple, Dict, TypedDict, cast
from enum import Enum

import psycopg

from ..Config.config import Settings
from .embedder import BGEM3F_Embed


class Corpus_Retrieval(TypedDict):
    chunk_id: int
    ner_entities: List[str]
    norm_doc: str
    compressed_doc: str
    metadata: Dict[str, int]


settings = Settings()


class Queries(Enum):
    """Enum for SQL queries called in Source dense retrieval."""

    RAW = f"""with q(v) as (values (%s::vector(1024))) select r.chunk_id, r.norm_embedding<=> q.v as cos_dist, norm_doc, \
            metadata from {settings.PG_DATABASE} as r cross join q order by cos_dist \
            limit (%s);"""
    COMPRESSED = f"""with q(v) as (values (%s::vector(1024))) select r.chunk_id, r.compressed_embedding <=> q.v as cos_dist, compressed_doc, \
            metadata from {settings.PG_DATABASE} as r cross join q order by cos_dist \
            limit (%s);"""
    NORM_CORPUS = f"""select chunk_id, ner_entities, norm_doc, compressed_doc, metadata from {settings.PG_DATABASE} order by chunk_id;"""


class Source:
    """Source class for retrieving queries from the database."""

    def __init__(self, config: Settings, embed_model: BGEM3F_Embed):
        """Creates a Source object.

        Args:
            config (Settings): Settings object.
            embed_model (BGEM3F_Embed): Embedding model object.

        Returns:
            Source: Source object.
        """
        self._config = config
        self.conn = psycopg.connect(f"dbname={self._config.PG_DATABASE}")
        self.embed_model = embed_model

    # Similarity search query - note vectors are normalized so cosine dist = 1 - dot_prod(x,y) thus take the smallest

    def close(self):
        """Closes the connection to the database."""
        self.conn.close()

    def dense_retrieval(
        self,
        query: str,
        raw_or_compressed: Literal["raw", "compressed"],
    ) -> List[Tuple[int, float, str, Dict[str, int]]]:
        """Dense retrieval from the database.

        Args:
            query (str): Query string.
            raw_or_compressed (Literal["raw", "compressed"]): Whether to use raw or LLM-compressed embeddings.

        Returns:
            List[Tuple[int, float, str, Dict[str, int]]]: List of tuples containing the chunk_id, cosine distance, norm_doc and metadata dictionary.
        """

        top_k = self._config.DENSE_TOP_K
        query_vect: List[float] = self.embed_model.embed_query(query)
        cur = self.conn.cursor()

        if raw_or_compressed == "raw":
            print("executed raw")
            sql_string = Queries.RAW.value
        else:
            print("executed compressed")
            sql_string = Queries.COMPRESSED.value

        cur.execute(sql_string, (query_vect, top_k))
        results = cur.fetchall()
        cur.close()

        return results

    def corpus_retrieval(self) -> List[Corpus_Retrieval]:
        """
        Retrieves the a full entry across from the database except for raw non-normed text.

        Returns:
            List[Corpus_Retrieval]: List of Corpus_Retrieval objects.
        """
        cur = self.conn.cursor()
        cur.execute(Queries.NORM_CORPUS.value)
        results = cur.fetchall()

        results_list: List[Corpus_Retrieval] = [
            cast(
                Corpus_Retrieval, dict(zip(Corpus_Retrieval.__annotations__.keys(), x))
            )
            for x in results
        ]

        cur.close()

        return results_list
