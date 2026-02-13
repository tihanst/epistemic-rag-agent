from typing import List, Tuple, Dict, Union, TypedDict
from functools import reduce
import sys

Idx2 = Tuple[int, int]
Idx3 = Tuple[int, int, int]
Span = Union[Idx2, Idx3]
ItemIn = Tuple[Span, str]
ItemOut = Tuple[Idx2, str]


class Data(TypedDict):
    chunk_id: int
    ner_entities: List[str]
    norm_doc: str
    compressed_doc: str
    metadata: Dict[str, int]


# Do not use, prefer to use linear implementation with no max_overlap parameter. This still works but hasn't been tested
def merge_overlap(
    p1: str, p2: str, max_overlap: int = 200, *, casefold: bool = False
) -> str:
    """
    Merge two paragraphs where p2 starts with text that also appears as a suffix
    of the last ≤ max_overlap characters of p1. If no overlap, just concatenates.

    Args:
        p1: First paragraph
        p2: Second paragraph
        max_overlap: Only consider this many trailing characters from p1
        casefold: If True, case-insensitive match (uses Unicode casefold)

    Returns:
        Merged paragraph (no duplicated overlap).
    """
    tail = p1[-max_overlap:]
    s_tail = tail.casefold() if casefold else tail
    s_p2 = p2.casefold() if casefold else p2

    # longest prefix of p2 that is a suffix of tail
    max_k = min(len(s_p2), len(s_tail))
    for k in range(max_k, 0, -1):
        if s_tail.endswith(s_p2[:k]):
            return p1 + " " + p2[k:]
    print("No match")
    return p1 + " " + p2


# Only use this linear implementation
def merge_overlap_linear(
    p1: str,
    p2: str,
    max_overlap: int | None = None,
    *,
    casefold: bool = False,
    sep: str = " ",
) -> str:
    """
    Merge two paragraphs where p2 starts with text that also appears as a suffix
    of the last ≤ max_overlap characters of p1. If no overlap, just concatenates.

    - Linear-time overlap detection via KMP prefix function.
    - If max_overlap=None, allows full-length overlap.
    - casefold=True for Unicode-insensitive matching (content merged from originals).
    - sep controls what is inserted between p1 and the non-overlapping tail of p2.
    """
    # Prepare match views (optionally case-insensitive), but keep originals for output.
    s1 = p1.casefold() if casefold else p1
    s2 = p2.casefold() if casefold else p2

    # Limit how much of p1’s suffix we consider
    if max_overlap is None:
        suffix_view = s1
    else:
        # Don’t consider more than p1’s length
        suffix_view = s1[-min(max_overlap, len(s1)) :]

    # We want the longest L such that suffix_view ends with s2[:L].
    # Standard trick: run KMP prefix-function on s2 + '#' + suffix_view
    # The last pi value is the length of the longest prefix of s2 that’s also a suffix of suffix_view.
    def prefix_function(t: str) -> list[int]:
        pi = [0] * len(t)
        j = 0
        for i in range(1, len(t)):
            while j > 0 and t[i] != t[j]:
                j = pi[j - 1]
            if t[i] == t[j]:
                j += 1
            pi[i] = j
        return pi

    # Use a separator not appearing in typical text; if unsure, pick a control char.
    # (No correctness risk if it appears—just slightly different pi path.)
    sep_char = "\x00"
    combo = s2 + sep_char + suffix_view
    pi = prefix_function(combo)
    L = pi[-1]  # longest overlap length

    # Construct merged output: avoid duplicate overlap.
    # Optional nicety: avoid double spaces when overlap boundary already provides whitespace.
    tail = p2[L:]
    if p1 and tail:
        # If p1 ends with whitespace or tail starts with punctuation/no-space-needed, you might skip sep.
        # Keep it simple: only add sep if neither side provides boundary whitespace.
        add_sep = not p1[-1:].isspace() and not tail[:1].isspace()
        return p1 + (sep if add_sep else "") + tail
    else:
        return p1 + tail


def expand_and_merge_linear(
    data: List[Tuple[int, str, Data]],
    corpus: List[str],
) -> List[Tuple[Tuple[int, int, int], str]]:
    """
    Function to take list of retrieved texts and pad with preceding and following text.

    Args:
        data: Retrieved relevant data from database with text, ner_entities, etc.
        corpus: Actual documents.

    Returns:
        List of tuples with the index span of text coverd, and the combined non-overlapping text.
    """


    sort_list = [(x[0], x[1]) for x in data]
    sort_list = sorted(sort_list, key=lambda x: x[0])
    assert type(sort_list) is list, (
        f"sort_list is not a list it is type {type(sort_list)}"
    )

    n = len(corpus)
    try:
        assert n != 0, "Corpus passed to expand_and_merge_linear is empty"
    except AssertionError as e:
        print(e)
        sys.exit(1)

    expanded_list: List[Tuple[Tuple[int, int, int], str]] = []

    for x in sort_list:
        idx = x[0]

        # Sanity check: incoming index must be within corpus
        if not (0 <= idx < n):
            raise IndexError(f"index {idx} is outside corpus of length {n}")

        if 0 < idx < n - 1:
            # has neighbors on both sides: (idx-1, idx, idx+1)
            merged = merge_overlap_linear(
                merge_overlap_linear(corpus[idx - 1], corpus[idx]),
                corpus[idx + 1],
            )
            expanded_list.append(((idx - 1, idx, idx + 1), merged))

        elif idx == 0:
            # left edge: need 0,1,2; require n >= 3
            if n >= 3:
                merged = merge_overlap_linear(
                    merge_overlap_linear(corpus[0], corpus[1]),
                    corpus[2],
                )
                expanded_list.append(((0, 1, 2), merged))
            elif n == 2:
                merged = merge_overlap_linear(corpus[0], corpus[1])
                expanded_list.append(((0, 0, 1), merged))  # pad a slot consistently
            else:  # n == 1
                expanded_list.append(((0, 0, 0), corpus[0]))

        else:  # idx == n - 1
            # right edge: need n-3,n-2,n-1; require n >= 3
            if n >= 3:
                merged = merge_overlap_linear(
                    merge_overlap_linear(corpus[n - 3], corpus[n - 2]),
                    corpus[n - 1],
                )
                expanded_list.append(((n - 3, n - 2, n - 1), merged))
            elif n == 2:
                merged = merge_overlap_linear(corpus[0], corpus[1])
                expanded_list.append(((0, 1, 1), merged))  # pad a slot consistently
            else:  # n == 1
                expanded_list.append(((0, 0, 0), corpus[0]))

    return expanded_list


def _is_mergeable(x: tuple[int, int, int], y: tuple[int, int, int]) -> bool:
    """
    Helper function to check if two triplets are mergeable.

    Args:
        x: First section index triplet.
        y: Second section index triplet.
    Returns:
        True if mergeable, False otherwise.
    """
    # Merge if overlapping or abutting: e.g. (4,5,6)-(5,6,7) or (4,5,6)-(7,8,9)
    return x[-1] >= y[0] - 1


def group_mergeable(
    content: list[tuple[tuple[int, int, int], str]],
) -> list[list[tuple[tuple[int, int, int], str]]]:
    """
    Function to group mergeable triplets.

    Args:
        content: List of (indices, text) tuples where indices are either 2-tuple (start,end) or 3-tuple (triplet).
    Returns:
        List of lists of mergeable (indices, text) tuples.
    """
    if not content:
        return []
    groups = []
    current_group = [content[0]]
    for i in range(1, len(content)):
        if _is_mergeable(content[i - 1][0], content[i][0]):
            current_group.append(content[i])
        else:
            groups.append(current_group)
            current_group = [content[i]]
    groups.append(current_group)
    return groups


# If group has one item: keep its triplet.
# If group has >1: output (start, end), where start is the first integer of the first triplet, end is the last integer of the last triplet.
def merge_group(
    group: list[tuple[tuple[int, int, int], str]],
) -> tuple[tuple[int, ...], str]:
    """
    Function to merge a group of mergeable triplets.

    Args:
        group: List of (indices, text) tuples where indices are either 2-tuple (start,end) or 3-tuple (triplet).
    Returns:
        Tuple of (indices, merged_text) where indices is either 2-tuple (start,end) or 3-tuple (triplet).
    """
    if len(group) == 1:
        # No merge, keep triplet
        indices = group[0][0]
    else:
        # Merge, use (first, last)
        indices = (group[0][0][0], group[-1][0][-1])
    merged_text = reduce(lambda a, b: merge_overlap_linear(a, b), [t for _, t in group])
    return indices, merged_text


def reduce_and_return(
    content: list[tuple[tuple[int, int, int], str]],
) -> list[tuple[tuple[int, ...], str]]:
    """
    Function to reduce a list of (indices, text) tuples where indices are either 2-tuple (start,end) or 3-tuple (triplet),
    and returns a list of (indices, merged_text) tuples where indices are either 2-tuple (start,end) or 3-tuple (triplet).

    Args:
        content: List of (indices, text) tuples where indices are either 2-tuple (start,end) or 3-tuple (triplet).
    Returns:
        List of (indices, merged_text) tuples where indices are either 2-tuple (start,end) or 3-tuple (triplet).

    """
    groups = group_mergeable(content)
    return [merge_group(g) for g in groups]


def clean_gap_one_merges(
    merged: List[Tuple[Tuple[int, ...], str]], *, merge_func=merge_overlap_linear
) -> List[Tuple[Tuple[int, ...], str]]:
    """
    Given a list of (indices, text) tuples where indices are either 2-tuple (start,end) or 3-tuple (triplet),
    merges together any consecutive pairs whose last index and first index have a gap of exactly 1.
    """
    if not merged:
        return []
    result = []
    cur_indices, cur_text = merged[0]
    for next_indices, next_text in merged[1:]:
        cur_end = cur_indices[-1]
        next_start = next_indices[0]
        if next_start - cur_end == 2:  # Gap of one (e.g., 4189-4191)
            # Merge!
            # Determine new indices:
            if len(cur_indices) == 2:
                new_start = cur_indices[0]
            else:
                new_start = cur_indices[0]
            if len(next_indices) == 2:
                new_end = next_indices[-1]
            else:
                new_end = next_indices[-1]
            # Merge text
            merged_text = merge_func(cur_text, next_text)
            cur_indices = (new_start, new_end)
            cur_text = merged_text
        else:
            # No merge, push current
            result.append((cur_indices, cur_text))
            cur_indices, cur_text = next_indices, next_text
    result.append((cur_indices, cur_text))
    return result


def thin_list_of_redundant(
    to_be_thinned: List[Tuple[Tuple[int, ...], str]],
    reference_list: List[Tuple[Tuple[int, ...], str]],
) -> List[Tuple[Tuple[int, ...], str]]:
    """
    Given a list of (indices, text) tuples where indices are either 2-tuple (start,end) or 3-tuple (triplet),
    removes any tuples whose text is already present in the reference list.

    Args:
        to_be_thinned: List of (indices, text) tuples to be thinned.
        reference_list: List of (indices, text) tuples to compare against.
    Returns:
        List of (indices, text) tuples with redundant text removed.

    """
    new_list = [x for x in to_be_thinned if x[1] not in [y[1] for y in reference_list]]
    return new_list
