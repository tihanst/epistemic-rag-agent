from typing import List, Tuple, Union, Dict, Set, Callable, Hashable
from collections import defaultdict

INF = 2**31 - 1


class RRF:
    """
    RRF class for RRF fusion
    """

    def __init__(self, K_param: int = 60) -> None:
        """
        Initialize RRF class
        Args:
            K_param (int, optional): K parameter for RRF. Defaults to 60.
        """
        self.K = K_param

    def rrf_fuse(
        self,
        rank_lists: List[List[int]],
        topk: Union[int, None] = None,
        # ensure deterministic final tie-break
        canonical_id: Callable[[Hashable], str] = lambda x: str(x),
    ) -> Tuple[List[Tuple[int, float]], List[int]]:
        """Fuse rank lists using RRF

        Args:
            rank_lists (List[List[int]]): A list of lists of document ids, where each list is a rank list of documents arranged from best to worst.
            topk (Union[int, None], optional) : The number of documents to return. Defaults to None meaning all documents are returned.
            canonical_id (str, optional): A function that takes a document id and returns a string that is used to break ties. Defaults to lambda x: str(x).

        Returns:
            Tuple[List[Tuple[int, float]], List[int]]: A tuple of two lists, the first list contains the document ids and their scores, the second list contains the document ids.
        """

        score: Dict[int, float] = defaultdict(float)
        K = self.K
        support: Dict[int, int] = defaultdict(int)  # in how many lists did d appear?
        best_rank: Dict[int, int] = defaultdict(lambda: INF)

        for L in rank_lists:
            seen: Set[int] = set()
            for r, d in enumerate(L, 1):  # lists are best→worst
                if d in seen:
                    continue  # harmless even if you guarantee no dups
                seen.add(d)
                score[d] += 1.0 / (K + r)
                if r < best_rank[d]:
                    best_rank[d] = r
            for d in seen:
                support[d] += 1

        items = list(score.items())
        items.sort(
            key=lambda kv: (
                -kv[1],  # 1) higher RRF
                -support[kv[0]],  # 2) more list support
                best_rank[kv[0]],  # 3) better (lower) best rank
                canonical_id(kv[0]),  # 4) stable final tie-break
            )
        )
        if topk is not None:
            items = items[:topk]
        return items, [d for d, _ in items]
