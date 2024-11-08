"""Common functions for operations on a standardized dataset"""
import random

def sample_multiple_entries(dataset, n=1):
    """Randomly select and return `n` entries from the dataset."""
    if not dataset:
        raise ValueError("The dataset is empty, cannot sample entries.")
    if n > len(dataset):
        raise ValueError(f"Requested {n} samples, but the dataset only has {len(dataset)} entries.")
    return random.sample(dataset, n)

