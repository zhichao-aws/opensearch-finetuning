"""
Data loading utilities for training
"""
import json
import logging
from pathlib import Path
from typing import List

from sentence_transformers import InputExample

logger = logging.getLogger(__name__)


def load_training_data(data_dir: str, max_samples: int = None) -> List[InputExample]:
    """
    Load training data from JSONL file.

    Expected format:
    {"query": "...", "document": "..."}

    Returns list of InputExample with (query, document) positive pairs.

    Args:
        data_dir: Directory containing training data
        max_samples: Maximum number of samples to load

    Returns:
        List of InputExample objects
    """
    training_file = Path(data_dir) / "training_data.jsonl"

    if not training_file.exists():
        raise FileNotFoundError(f"Training file not found: {training_file}")

    logger.info(f"Loading training data from: {training_file}")

    examples = []

    with open(training_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if max_samples and idx >= max_samples:
                break

            try:
                data = json.loads(line.strip())
                query = data["query"]
                document = data["document"]

                # Create InputExample for contrastive learning
                # texts=[query, document] creates a positive pair
                example = InputExample(texts=[query, document])
                examples.append(example)

            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Skipping invalid line {idx}: {e}")
                continue

    logger.info(f"Loaded {len(examples)} valid training examples")
    return examples
