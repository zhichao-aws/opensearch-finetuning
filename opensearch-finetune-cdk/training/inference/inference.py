"""
SageMaker Inference Handler for Fine-Tuned Retrieval Models
Handles model loading and inference for sentence-transformers models
"""
import json
import logging
import os
from typing import Dict, List, Union

import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Global model variable
model = None


def model_fn(model_dir: str) -> SentenceTransformer:
    """
    Load the model for inference.
    Called once when the endpoint is initialized.

    Args:
        model_dir: Directory containing the model artifacts

    Returns:
        Loaded SentenceTransformer model
    """
    global model

    logger.info(f"Loading model from {model_dir}")

    try:
        model = SentenceTransformer(model_dir)

        # Use GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        logger.info(f"Model loaded successfully on device: {device}")

        # Log model info
        if hasattr(model, 'get_sentence_embedding_dimension'):
            dim = model.get_sentence_embedding_dimension()
            logger.info(f"Embedding dimension: {dim}")

        return model

    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}", exc_info=True)
        raise


def input_fn(request_body: str, content_type: str) -> Dict:
    """
    Parse input data for inference.

    Args:
        request_body: The request body as a string
        content_type: The content type of the request

    Returns:
        Parsed input data as dictionary

    Raises:
        ValueError: If content type is not supported
    """
    if content_type == "application/json":
        try:
            input_data = json.loads(request_body)
            return input_data
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON input: {str(e)}")
            raise ValueError(f"Invalid JSON input: {str(e)}")
    else:
        raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_data: Dict, model: SentenceTransformer) -> Dict:
    """
    Generate embeddings for input text.

    Expected input formats:
    1. {"inputs": "single text"}
    2. {"inputs": ["text1", "text2", ...]}
    3. {"text_inputs": ["text1", "text2", ...]}

    Args:
        input_data: Dictionary containing input texts
        model: The loaded SentenceTransformer model

    Returns:
        Dictionary with embeddings
    """
    logger.info("Running inference...")

    try:
        # Extract text inputs
        if "inputs" in input_data:
            texts = input_data["inputs"]
        elif "text_inputs" in input_data:
            texts = input_data["text_inputs"]
        else:
            raise ValueError("Input must contain 'inputs' or 'text_inputs' field")

        # Handle single text or list of texts
        if isinstance(texts, str):
            texts = [texts]

        if not isinstance(texts, list):
            raise ValueError("Input texts must be a string or list of strings")

        logger.info(f"Generating embeddings for {len(texts)} text(s)")

        # Generate embeddings
        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32
        )

        logger.info(f"Generated embeddings with shape: {embeddings.shape}")

        return {"embeddings": embeddings.tolist()}

    except Exception as e:
        logger.error(f"Inference failed: {str(e)}", exc_info=True)
        raise


def output_fn(prediction: Dict, accept: str) -> str:
    """
    Serialize prediction output.

    Args:
        prediction: Dictionary containing predictions
        accept: The accept header from the request

    Returns:
        Serialized prediction output

    Raises:
        ValueError: If accept type is not supported
    """
    if accept == "application/json" or accept.startswith("application/json"):
        return json.dumps(prediction)
    else:
        raise ValueError(f"Unsupported accept type: {accept}")


# Alternative: Combined transform_fn for full control
def transform_fn(model: SentenceTransformer, input_data: bytes, content_type: str, accept: str) -> tuple:
    """
    Alternative to using separate input_fn, predict_fn, output_fn.
    Provides full control over the inference pipeline.

    Args:
        model: The loaded model
        input_data: Raw input data
        content_type: Content type of input
        accept: Accept header

    Returns:
        Tuple of (response_body, content_type)
    """
    # Parse input
    input_dict = input_fn(input_data.decode('utf-8'), content_type)

    # Run prediction
    prediction = predict_fn(input_dict, model)

    # Serialize output
    output = output_fn(prediction, accept)

    return output, accept
