import torch


def find_common_subsequences(seq1, seq2, min_length=1):
    """
    Find all common subsequences of minimum length between two sequences using PyTorch
    """
    device = seq1.device

    common_subsequences = []
    n, m = len(seq1), len(seq2)

    # Create tensor to track lengths of common subsequences
    lengths = torch.zeros((n + 1, m + 1), dtype=torch.int64).to(device=device)

    # Fill the tensor
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if seq1[i - 1] == seq2[j - 1]:
                lengths[i, j] = lengths[i - 1, j - 1] + 1

                if lengths[i, j] >= min_length:
                    length = int(lengths[i, j].item())
                    subseq = seq1[i - length:i].clone().detach()

                    # Check if this subsequence is already in our list
                    is_new = True
                    for existing_subseq, _, _, _, _ in common_subsequences:
                        if len(existing_subseq) == len(subseq) and all(
                                subseq[k] == existing_subseq[k] for k in range(len(subseq))):
                            is_new = False
                            break

                    if is_new:
                        common_subsequences.append(
                            (subseq.tolist(), i - length, i - 1, j - length, j - 1)
                        )

    # Sort by length (longest first)
    common_subsequences.sort(key=lambda x: len(x[0]), reverse=True)
    return common_subsequences


def subsequence_similarity(seq1, seq2, common_seqs=None):
    """
    Calculate similarity measure based on common subsequences using PyTorch
    """
    # Find common subsequences if not provided
    if common_seqs is None:
        common_seqs = find_common_subsequences(seq1, seq2, min_length=2)

    if not common_seqs:
        return torch.tensor(1.0, device=seq1.device)  # No common subsequences found

    # Sum the lengths of all common subsequences, but be careful about overlapping
    covered_indices_seq1 = set()
    covered_indices_seq2 = set()

    # Sort by length to prioritize longer subsequences
    sorted_seqs = sorted(common_seqs, key=lambda x: len(x[0]), reverse=True)

    total_matching = 0 # Initialize total matching count

    for subseq, start1, end1, start2, end2 in sorted_seqs:
        # Add indices to covered sets if they're not already covered
        seq1_indices = set(range(start1, end1 + 1))
        seq2_indices = set(range(start2, end2 + 1))

        # Only count non-overlapping indices
        new_indices_seq1 = seq1_indices - covered_indices_seq1
        new_indices_seq2 = seq2_indices - covered_indices_seq2

        covered_indices_seq1.update(new_indices_seq1)
        covered_indices_seq2.update(new_indices_seq2)

        total_matching += len(new_indices_seq1) ** 2

    # Get the length of the longer sequence
    max_len = max(len(seq1), len(seq2))
    total_possible_matching = max_len ** 2

    # Calculate similarity score
    # 0 if sequences are identical, 1 if no common subsequences
    similarity = 1.0 - (total_matching / total_possible_matching)

    return torch.tensor(similarity, device=seq1.device)