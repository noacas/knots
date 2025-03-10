from smart_collapse import remove_consecutive_inverses, remove_free_strands, destabilize, remove_nonconsecutive_inverses

import torch

def test_remove_consecutive_inverses():
    braid = torch.tensor([1, -1, 2, 3], dtype=torch.float)
    assert remove_consecutive_inverses(braid).equal(torch.tensor([2, 3], dtype=torch.float))
    braid = torch.tensor([1, -1, 2, -2, 3], dtype=torch.float)
    assert remove_consecutive_inverses(braid).equal(torch.tensor([3], dtype=torch.float))
    braid = torch.tensor([1, 2, -1], dtype=torch.float)
    assert remove_consecutive_inverses(braid).equal(torch.tensor([1, 2, -1], dtype=torch.float))


def test_remove_free_strands():
    braid = torch.tensor([1, 3], dtype=torch.float)
    assert remove_free_strands(braid).equal(torch.tensor([1,3], dtype=torch.float))
    braid = torch.tensor([1, 2, 1], dtype=torch.float)
    assert remove_free_strands(braid).equal(torch.tensor([1,2,1], dtype=torch.float))
    braid = torch.tensor([2, 2], dtype=torch.float)
    assert remove_free_strands(braid).equal(torch.tensor([1,1], dtype=torch.float))
    braid = torch.tensor([1, 4], dtype=torch.float)
    assert remove_free_strands(braid).equal(torch.tensor([1,3], dtype=torch.float))


def test_destabilize():
    braid = torch.tensor([1, 2, -2, 3], dtype=torch.float)
    assert destabilize(braid).equal(torch.tensor([1, 2, -2], dtype=torch.float))
    braid = torch.tensor([1, 2, 3], dtype=torch.float)
    assert destabilize(braid).equal(torch.tensor([1, 2], dtype=torch.float))
    braid = torch.tensor([-3, 2, 1, 3], dtype=torch.float)
    assert destabilize(braid).equal(torch.tensor([-3, 2, 1, 3], dtype=torch.float))


def test_remove_nonconsecutive_inverses():
    braid = torch.tensor([1, 2, -1, 3], dtype=torch.float)
    assert remove_nonconsecutive_inverses(braid).equal(torch.tensor([1, 2, -1, 3], dtype=torch.float))
    braid = torch.tensor([1, 3, -1, 2], dtype=torch.float)
    assert remove_nonconsecutive_inverses(braid).equal(torch.tensor([3, 2], dtype=torch.float))
