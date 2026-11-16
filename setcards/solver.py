"""SET-Regel: drei Karten bilden ein Set, wenn jedes Attribut bei allen gleich oder bei allen verschieden ist.

Karten als Tupel von 4 Klassenindizes (0..2). Gleich oder alle verschieden <=> Summe % 3 == 0.
"""
from __future__ import annotations

from itertools import combinations
from typing import Sequence


def is_set(a: Sequence[int], b: Sequence[int], c: Sequence[int]) -> bool:
    return all((x + y + z) % 3 == 0 for x, y, z in zip(a, b, c))


def third_card(a: Sequence[int], b: Sequence[int]) -> tuple[int, ...]:
    """Die eine Karte, die a und b zu einem Set ergänzt."""
    return tuple((-(x + y)) % 3 for x, y in zip(a, b))


def find_sets(cards: Sequence[Sequence[int]]) -> list[tuple[int, int, int]]:
    """Alle Sets als Indextripel (i < j < k). Doppelte Karten werden normal behandelt."""
    pos: dict[tuple[int, ...], list[int]] = {}
    for i, c in enumerate(cards):
        pos.setdefault(tuple(c), []).append(i)
    out = set()
    for i, j in combinations(range(len(cards)), 2):
        for k in pos.get(third_card(cards[i], cards[j]), []):
            if k != i and k != j:
                out.add(tuple(sorted((i, j, k))))
    return sorted(out)
