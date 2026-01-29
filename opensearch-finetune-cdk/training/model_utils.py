"""
Model initialization utilities
"""
import logging
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def initialize_model(
    model_id: str,
    model_type: str,
    max_seq_length: int = 512
) -> SentenceTransformer:
    """
    Initialize SentenceTransformer model.

    Args:
        model_id: HuggingFace model ID (e.g., "BAAI/bge-base-en-v1.5")
        model_type: "dense" or "sparse"
        max_seq_length: Maximum sequence length

    Returns:
        SentenceTransformer model ready for training
    """

    logger.info(f"Loading base model: {model_id}")

    # Load pre-trained model
    model = SentenceTransformer(model_id)

    # Set max sequence length
    model.max_seq_length = max_seq_length

    # For sparse models, ensure proper tokenization
    if model_type == "sparse":
        logger.info("Configuring for sparse model training...")
        # Sparse models in sentence-transformers typically use special
        # pooling strategies or output layers
        # This may require custom implementation based on the architecture

    logger.info(f"Model initialized with max_seq_length={max_seq_length}")

    return model
