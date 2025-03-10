import random

import torch

"""
If two closed braids represent the same ambient isotopy class of oriented links (cf. also Braid theory), then one can transform one braid to another by a sequence of Markov moves:
i) a↔bab^(−1) (conjugation).
ii) a↔ab^(±1) where a is an element of the n-th braid group and b is in the n+1-th braid group (adding a new strand).
"""


def max_abs_braid(braid: torch.Tensor) -> int:
    return max(abs(x) for x in braid)


def conjugation_markov_move(braid: torch.Tensor, j: int, k: int) -> torch.Tensor:
    """
    Implements the Conjugation Markov Move algorithm.
    Returns the braid [(−1)^k j] + braid + [(−1)^(k+1) j]
    k = 0 or 1
    j = 1, 2, ..., max(abs(braid))
    """
    # Calculate the terms to add
    term1 = torch.tensor([(-1) ** k * j], dtype=torch.float16)
    term2 = torch.tensor([(-1) ** (k + 1) * j], dtype=torch.float16)
    # Update braid: braid ← [(−1)^k j] + braid + [(−1)^(k+1) j]
    return torch.cat((term1, braid, term2), dim=0)


def random_conjugation_markov_move(braid: torch.Tensor) -> torch.Tensor:
    # Random index between 1 and max(abs(braid))
    j = random.randint(1, max_abs_braid(braid))
    # Random choice between 0 and 1 for k
    k = random.randint(0, 1)
    return conjugation_markov_move(braid, j, k)


def new_strand_markov_move(braid: torch.Tensor, k: int) -> torch.Tensor:
    """
    Implements the New Strand Markov Move algorithm.
    Returns the braid: braid + [(−1)^k (max(abs(braid))+1)]
    k = 0 or 1
    """
    # Calculate the term to add
    term = (-1) ** k * (max_abs_braid(braid) + 1)
    # Update braid: braid ← braid + [(−1)^k (max(abs(braid))+1)]
    new_element = torch.tensor([term], dtype=torch.float16)
    return torch.cat((braid, new_element))


def random_new_strand_markov_move(braid: torch.Tensor) -> torch.Tensor:
    # Random choice between 0 and 1 for k
    k = random.randint(0, 1)
    return new_strand_markov_move(braid, k)


def random_markov_move(braid: torch.Tensor, with_new_strand=True) -> torch.Tensor:
    """
    Implements the Random Markov Move algorithm.
    """
    # Random choice between conjugation (0) and new strand (1) move
    i = random.randint(0, 1)
    if i == 0 or not with_new_strand:  # Conjugation Markov move
        braid = random_conjugation_markov_move(braid)

    else:  # New strand Markov move
        braid = random_new_strand_markov_move(braid)

    return braid
