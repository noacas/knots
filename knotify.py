import random
import torch

from components import get_n_strands, get_components


def knotify(braid: torch.Tensor) -> torch.Tensor:
    """
    Turn braid representative of a link into a knot. Iteratively weaves together two strands not in the same link component until the braid closure is a knot
    Notice the returned braid is different from the input braid (because the input braid may not be a knot)
    """
    components = get_components(braid)
    while braid.numel() > 0 and len(components) != 1:
        strands = set(range(1, get_n_strands(braid) + 1))
        to_add = 0
        for component in components:
            for strand in component:
                up, down = strand + 1, strand - 1
                i = random.randint(0, 1)
                if up not in component and up in strands:
                    to_add = (-1) ** i * strand  # weave together strand, strand +1
                    break
                elif down not in component and down in strands:
                    to_add = (-1) ** i * (strand - 1)
                    break
            if to_add != 0:
                term = torch.tensor([to_add], dtype=torch.float)
                braid = torch.cat((braid, term))
                break
        components = get_components(braid)
    return braid
