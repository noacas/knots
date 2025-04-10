import torch

def _braid_relation1(braid: torch.Tensor, i: int) -> (torch.Tensor, bool):
    """
    Apply braid relation 1 to the braid at index i if possible
    """
    device = braid.device
    length = braid.numel()
    p1, p2, p3 = i % length, (i + 1) % length, (i + 2) % length
    # Check for pattern [±k, ±(k+1), ±k] or [±(k+1), ±k, ±(k+1)]
    if braid[p1] == braid[p3] and (braid[p2] == braid[p1] + 1 or braid[p2] == braid[p1] - 1):
        # Check for pattern [k, (k+1), k] or [-(k+1), -k, -(k+1)] or [(k+1), k, (k+1)] or [-k, -(k+1), -k]
        if device != "cpu":
            braid = braid.cpu()
        indices = torch.tensor([p1, p2, p3])
        values = braid[torch.tensor([p2, p1, p2])].cpu()
        braid.index_copy_(0, indices, values)
        if device != "cpu":
            braid = braid.to(device)
        return braid, True
    return braid, False


def braid_relation1(braid: torch.Tensor) -> torch.Tensor:
    """
    Apply braid relation 1 to the braid at first opportunity and then shift the result to the right (the replaced elements will be last in the braid)
    """
    for i in range(len(braid)):
        braid, changed = _braid_relation1(braid, i)
        if changed:
            return torch.concat([braid[i + 3:], braid[:i], braid[i:i + 3]], dim=0)
    return braid


def braid_relation1_on_entire_braid(braid: torch.Tensor, start=0, take_closure=True) -> torch.Tensor:
    """
    Implements the BraidRelation1 algorithm for braid manipulation. σ1σ2σ1 = σ2σ1σ2
    """
    i = start
    while i < start + braid.shape[0]:
        if not take_closure and i >= braid.shape[0] - 2:
            break
        braid, changed = _braid_relation1(braid, i)
        if changed:
            i += 3
        else:
            i += 1
    return braid


def braid_relation2(braid: torch.Tensor) -> torch.Tensor:
    """
    Apply braid relation 2 to the braid at first opportunity and then shift the result to the right.
    The replaced elements will be last in the braid.
    """
    if braid.numel() == 0:
        return braid
    prev = braid[0]
    for i, cur in enumerate(braid[1:]):
        if abs(abs(prev) - abs(cur)) >= 2:
            return torch.cat((braid[i+2:], braid[:i], torch.tensor([cur, prev], dtype=braid.dtype, device=braid.device)), dim=0)
        prev = cur
    return braid


def shift_right(braid: torch.Tensor) -> torch.Tensor:
    """
    Shift the braid to the right by one
    """
    return torch.cat((braid[-1:], braid[:-1]))


def shift_left(braid: torch.Tensor) -> torch.Tensor:
    """
    Shift the braid to the left by one
    """
    return torch.cat((braid[1:], braid[:1]))
