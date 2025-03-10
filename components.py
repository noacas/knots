from typing import List, Set

import torch

def get_n_strands(braid: torch.Tensor) -> int:
    """Returns the number of strands in the braid."""
    return int(braid.abs().max().item()) + 1 if braid.numel() > 0 else 1

def get_components(braid: torch.Tensor) -> List[Set[int]]:
    """
    Given a braid representation, return the connected components of strands.
    """
    n_strands = get_n_strands(braid)
    components = [{i} for i in range(1, n_strands + 1)]

    for crossing in braid:
        i = abs(crossing.item())  # Extract integer from tensor
        comp1 = next((c for c in components if i in c), None)
        comp2 = next((c for c in components if i + 1 in c), None)

        if comp1 is not None and comp2 is not None and comp1 != comp2:
            components.remove(comp1)
            components.remove(comp2)
            components.append(comp1 | comp2)  # Merge sets

    return components
