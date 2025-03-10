import random
import torch

from knotify import knotify
from markov_move import random_markov_move
from smart_collapse import smart_collapse
from braid_relation import braid_relation1_on_entire_braid


def apply_random_markov_move_and_relations(braid: torch.Tensor, n_moves: int,
                                           max_length: int, with_new_strand: bool = True) -> torch.Tensor:
    last_valid_braid = braid
    # Apply random moves and relations
    for _ in range(n_moves):
        braid = random_markov_move(braid, with_new_strand)
        # Apply braid relation1 at random position
        start = random.randint(0, len(braid)-1)
        braid = braid_relation1_on_entire_braid(braid, start)
        collapsed_braid = smart_collapse(braid)
        if len(collapsed_braid) <= max_length:
            last_valid_braid = collapsed_braid
    return last_valid_braid


def random_knot(n_letters: int, n_strands: int, n_moves: int) -> torch.Tensor:
    """
    Generates a random knot representative.

    Args:
        n_letters: Desired length of the braid word
        n_strands: Number of strands to use
        n_moves: Number of random moves to apply

    Returns:
        A braid representation of a random knot
    """
    braid = torch.empty(0, dtype=torch.float) # Empty braid word

    while len(braid) != n_letters:
        # Reset if braid becomes too long
        if len(braid) > n_letters:
            braid = torch.empty(0, dtype=torch.float)

        # Build braid to desired length
        while len(braid) < n_letters:
            # Random choice between positive and negative crossing
            sign = (-1) ** random.randint(0, 1)
            # Random strand selection
            j = random.randint(1, n_strands-1)
            # Add crossing with appropriate sign
            new_element = torch.tensor([sign * j], dtype=torch.float)
            braid = torch.cat((braid, new_element))

        braid = knotify(braid)

        if len(braid):
            braid = apply_random_markov_move_and_relations(braid, n_moves, n_letters, with_new_strand=True)

    return braid


def two_random_equivalent_knots(n_max: int, n_max_second: int, n_moves: int) -> (torch.Tensor, torch.Tensor):
    """
    Generates two random knot representatives that are equivalent.
    """
    # Generate a random knot
    knot1 = random_knot(n_max, n_max, n_moves)

    # Apply random moves and relations to knot1 to get knot2
    knot2 = knot1.clone()
    knot2 = apply_random_markov_move_and_relations(knot2, n_moves, n_max_second)

    return knot1, knot2


# Example usage and testing
def test_random_knot():
    # Test case 1: Small knot
    print("Test 1 - Small knot:")
    result1 = random_knot(n_letters=6, n_strands=4, n_moves=5)
    print(f"Generated knot: {result1}")

    # Test case 2: Larger knot
    print("\nTest 2 - Larger knot:")
    result2 = random_knot(n_letters=10, n_strands=6, n_moves=8)
    print(f"Generated knot: {result2}")

    # Test case 3: Very simple knot
    print("\nTest 3 - Simple knot:")
    result3 = random_knot(n_letters=4, n_strands=3, n_moves=3)
    print(f"Generated knot: {result3}")

    # Test case 4: Two equivalent knots
    print("\nTest 4 - Two equivalent knots:")
    knot1, knot2 = two_random_equivalent_knots(n_max=10, n_max_second=20, n_moves=8)
    print(f"Knot 1: {knot1}")
    print(f"Knot 2: {knot2}")


if __name__ == "__main__":
    test_random_knot()
