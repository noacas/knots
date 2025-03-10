from braid_relation import braid_relation1_on_entire_braid, braid_relation1, \
    braid_relation2

import torch


def test_braid_relation1():
    # Test case 1: Pattern [k, k+1, k]
    braid = torch.tensor(data=[2, 3, 2, 1], dtype=torch.float16)
    assert braid_relation1_on_entire_braid(braid, 0, False).equal(torch.tensor(data=[3, 2, 3, 1], dtype=torch.float16))
    # Test case 2: Pattern [k+1, k, k+1]
    braid = torch.tensor(data=[3, 2, 3, 1], dtype=torch.float16)
    assert braid_relation1_on_entire_braid(braid, 0, False).equal(torch.tensor(data=[2, 3, 2, 1], dtype=torch.float16))
    # Test case 3: With negative numbers
    braid = torch.tensor(data=[-2, 3, -2, 1], dtype=torch.float16)
    assert braid_relation1_on_entire_braid(braid, 0, False).equal(torch.tensor(data=[-2, 3, -2, 1], dtype=torch.float16))
    braid = torch.tensor(data=[-2, -3, -2, 1], dtype=torch.float16)
    assert braid_relation1_on_entire_braid(braid, 0, True).equal(torch.tensor(data=[-3, -2, -3, 1], dtype=torch.float16))
    # Test case 4: With take_closure
    braid = torch.tensor(data=[2, 2, 1], dtype=torch.float16)
    assert braid_relation1_on_entire_braid(braid, 0, False).equal(torch.tensor(data=[2, 2, 1], dtype=torch.float16))
    braid = torch.tensor(data=[2, 2, 1], dtype=torch.float16)
    assert braid_relation1_on_entire_braid(braid, 0, True).equal(torch.tensor(data=[1, 1, 2], dtype=torch.float16))
    # Test case 5: long braid
    braid = torch.tensor(data=[1, 2, 3, 2, 1, 2, 3, 2, 1], dtype=torch.float16)
    assert braid_relation1_on_entire_braid(braid, 0, False).equal(torch.tensor(data=[1, 3, 2, 3, 1, 3, 2, 3, 1], dtype=torch.float16))
    braid = torch.tensor(data=[1, 2, 3, 2, 1, 2, 3, 2, 1, 2], dtype=torch.float16)
    assert braid_relation1_on_entire_braid(braid, 0, True).equal(torch.tensor(data=[2, 3, 2, 3, 1, 3, 2, 3, 2, 1], dtype=torch.float16))


def test_braid_relation1_and_shift_right():
    braid = torch.tensor(data=[2, 3, 2], dtype=torch.float16)
    assert braid_relation1(braid).equal(torch.tensor(data=[3, 2, 3], dtype=torch.float16))
    braid = torch.tensor(data=[1, 2, 4, 2, 3, 2, 4, 5, 4], dtype=torch.float16)
    assert braid_relation1(braid).equal(torch.tensor(data=[4, 5, 4, 1, 2, 4, 3, 2, 3], dtype=torch.float16))


def test_braid_relation2_and_shift_right():
    braid = torch.tensor(data=[-1, 2, 4, 2, 3, 2, 4, 5, 4], dtype=torch.float16)
    assert braid_relation2(braid).equal(torch.tensor(data=[2, 3, 2, 4, 5, 4, -1, 4, 2], dtype=torch.float16))
    braid = torch.tensor(data=[1, -3, 4, 2, 3, 2, 4, 5, 4, 5], dtype=torch.float16)
    assert braid_relation2(braid).equal(torch.tensor(data=[4, 2, 3, 2, 4, 5, 4, 5, -3, 1], dtype=torch.float16))
    braid = torch.tensor(data=[1, 2, 3, 4], dtype=torch.float16)
    assert braid_relation2(braid).equal(torch.tensor(data=[1, 2, 3, 4], dtype=torch.float16))
