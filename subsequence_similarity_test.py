import torch

from subsequence_similarity import subsequence_similarity

def test_subsequence_similarity():
    """Test the subsequence similarity function with various cases
    """
    # Test case 1: Identical Sequences
    seq1 = torch.tensor([1, 2, 3, 4, 5])
    seq2 = torch.tensor([1, 2, 3, 4, 5])
    result = subsequence_similarity(seq1, seq2)
    assert result.item() == 0, f"Expected 0, got {result.item()}"

    # Test case 2: Partially Overlapping
    seq1 = torch.tensor([1, 2, 3, 4, 5, 6])
    seq2 = torch.tensor([3, 4, 5, 6, 7, 6])
    result = subsequence_similarity(seq1, seq2)
    assert 0.5 < result.item() < 0.6, f"Expected between 0.5 and 0.6, got {result.item()}"

    # Test case 3: No Overlap
    seq1 = torch.tensor([1, 2, 3, 4])
    seq2 = torch.tensor([2, 5, 4])
    result = subsequence_similarity(seq1, seq2)
    assert result.item() == 1, f"Expected 1, got {result.item()}"

    # Test case 4: Multiple Common Subsequences
    seq1 = torch.tensor([1, 2, 3, 4, 1, 2])
    seq2 = torch.tensor([4, 1, 2, 1, 2, 3])
    result = subsequence_similarity(seq1, seq2)
    assert result.item() == 0.5, f"Expected 0.5, got {result.item()}"