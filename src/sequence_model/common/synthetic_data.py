"""Generate synthetic data for sequence_model training and testing."""

import pickle
import random
from pathlib import Path
from typing import List


def generate_synthetic_text_data(
    num_sequences: int = 100, sequence_length: int = 20
) -> List[List[str]]:
    """
    Generate synthetic text data as token sequences.

    Args:
        num_sequences: Number of sequences to generate
        sequence_length: Length of each sequence

    Returns:
        List of token sequences (each sequence is a list of strings)
    """
    # Simple vocabulary of common words
    vocabulary = [
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
        "cat", "runs", "walks", "sleeps", "eats", "drinks", "plays",
        "big", "small", "fast", "slow", "red", "blue", "green", "yellow",
        "happy", "sad", "good", "bad", "one", "two", "three", "four"
    ]

    sequences = []
    for _ in range(num_sequences):
        sequence = [random.choice(vocabulary) for _ in range(sequence_length)]
        sequences.append(sequence)

    return sequences


def save_synthetic_data(output_path: str, num_sequences: int = 100, sequence_length: int = 20) -> None:
    """
    Generate and save synthetic training data to pickle file.

    Args:
        output_path: Path where to save the pickled data
        num_sequences: Number of sequences to generate
        sequence_length: Length of each sequence
    """
    # Ensure output directory exists
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate synthetic data
    data = generate_synthetic_text_data(num_sequences=num_sequences, sequence_length=sequence_length)

    # Save to pickle file
    with open(output_path, "wb") as f:
        pickle.dump(data, f)

    print(f"Synthetic data saved to {output_path}")
