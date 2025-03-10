import torch

from markov_move import random_conjugation_markov_move, random_new_strand_markov_move


def test_conjugation_markov_move():
    braid = torch.tensor([-1, 2, 4, -3, 1], dtype=torch.float16)
    new_braid = random_conjugation_markov_move(braid.clone())
    assert len(new_braid) == len(braid) + 2
    assert new_braid[0] == - 1 * new_braid[-1]
    assert new_braid[1:-1].equal(braid)
    assert abs(new_braid[0]) <= max(abs(x) for x in braid)
    assert abs(new_braid[0]) > 0


def test_new_strand_markov_move():
    braid = torch.tensor([-1, 2, 4, -3, 1], dtype=torch.float16)
    new_braid = random_new_strand_markov_move(braid.clone())
    assert len(new_braid) == len(braid) + 1
    assert new_braid[:-1].equal(braid)
    assert abs(new_braid[-1]) == max(abs(x) for x in braid) + 1
    assert abs(new_braid[-1]) > 0
