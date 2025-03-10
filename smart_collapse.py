import torch

from components import get_n_strands


def remove_consecutive_inverses(braid: torch.Tensor) -> torch.Tensor:
    """
    Removes consecutive inverse elements from the braid.
    Example: [1, -1, 2, 3] -> [2, 3]
    """
    if not len(braid):
        return braid

    result = torch.empty(0, dtype=torch.float)
    i = 0
    while i < len(braid):
        # If we have a next element, and it's the inverse of current
        if i + 1 < len(braid) and braid[i] == -braid[i + 1]:
            i += 2  # Skip both elements
        else:
            term = torch.tensor([braid[i]], dtype=torch.float)
            result = torch.cat((result, term), dim=0)
            i += 1
    return result


def remove_free_strands(braid: torch.Tensor) -> torch.Tensor:
    """
    Removes free strands from a braid word.
    A strand is free if it doesn't cross with any other strands,
    meaning the index i and its adjacent value (i+1) never appear in the generators.

    Example:
        [1,3] -> [1,3]  # No free strands
        [1,2,1] -> [1,2,1]  # No free strands
        [2,2] -> [1,1]  # After removing free strand, indices are adjusted
        [1,4] -> [1,3]  # After removing free strand, indices are adjusted
    """
    if not len(braid):
        return braid

    # Get the maximum index used (the number of strands is max + 1)
    n_strands = get_n_strands(braid)

    # Track which indices are involved in crossings
    involved_strands = set()
    for crossing in braid:
        i = int(abs(crossing).item())
        involved_strands.add(i)
        involved_strands.add(i + 1)

    # Find free strands (strands that don't participate in any crossing)
    free_strands = set(range(1, n_strands)) - involved_strands

    if not free_strands:
        return braid

    # Create a mapping for the new indices after removing free strands
    mapping = {}
    next_index = 1
    for i in range(1, n_strands):
        if i not in free_strands:
            mapping[i] = next_index
            next_index += 1

    # Create new braid word with adjusted indices
    new_braid = torch.clone(braid)
    for j, crossing in enumerate(braid):
        i = int(abs(crossing.item()))
        new_i = mapping[i]
        new_braid[j] = torch.tensor([new_i], dtype=torch.float) if crossing > 0 \
                else torch.tensor([-new_i], dtype=torch.float)

    return new_braid


def destabilize(braid: torch.Tensor) -> torch.Tensor:
    """
    Implements the destabilization move: wσn → w if n = max(|σ|) + 1
    Returns: A new BraidWord with the destabilization applied if possible
    """
    if not len(braid) or len(braid) < 2:
        return braid

    sigma = abs(braid[-1]).item()
    w_braid_group = max([abs(sigma) for sigma in braid[:-1]]).item()
    if w_braid_group + 1 == sigma:
        return braid[:-1]
    return braid


def remove_nonconsecutive_inverses(braid: torch.Tensor) -> torch.Tensor:
    """
    Removes inverse elements that are not consecutive but can be made consecutive
    through valid braid moves.
    """
    if braid.numel() == 0:
        return braid

    def can_move_together(index_i, index_j):
        """Check if elements at positions i and j can be moved together."""
        # Elements can be moved if they don't interact with elements between them (braid relation 2)
        strand_i_1 = abs(braid[index_i])
        strand_i_2 = abs(braid[index_i]) + 1
        for k in range(index_i + 1, index_j):
            strand_k_1 = abs(braid[k])
            strand_k_2 = abs(braid[k]) + 1
            if strand_k_1 in [strand_i_1, strand_i_2] or strand_k_2 in [strand_i_1, strand_i_2]:
                return False
        return True

    result = braid.clone()
    i = 0
    while i < result.shape[0]:
        found_pair = False
        for j in range(i + 1, result.shape[0]):
            if result[i] == -result[j] and can_move_together(i, j):
                # Use masking to remove elements instead of `pop()`
                mask = torch.ones(result.shape[0], dtype=torch.bool)
                mask[i] = False
                mask[j] = False
                result = result[mask]  # Keep only the elements where mask is True
                found_pair = True
                break  # Restart search after removal
        if not found_pair:
            i += 1

    return result


def remove_inverses(braid: torch.Tensor) -> torch.Tensor:
    braid = remove_consecutive_inverses(braid)
    braid = remove_nonconsecutive_inverses(braid)
    return braid


def smart_collapse(braid: torch.Tensor) -> torch.Tensor:
    """
    Implements the SmartCollapse algorithm to reduce braid length.

    Args:
        braid: Input braid represented as a list of integers

    Returns:
        Reduced braid after applying all reduction methods
    """
    if not len(braid):
        return braid # Empty braid

    braid_prime = torch.empty(0, dtype=torch.float)  # Empty braid word

    while braid_prime.shape != braid.shape or not torch.equal(braid_prime, braid):
        braid_prime = braid.clone()

        # Apply all reduction methods
        braid = remove_consecutive_inverses(braid)
        braid = remove_free_strands(braid)
        braid = destabilize(braid)
        braid = remove_nonconsecutive_inverses(braid)

    return braid
