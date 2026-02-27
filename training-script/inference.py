"""
SageMaker Inference Handler for Fine-tuned Retrieval Models

Provides model_fn, input_fn, predict_fn, output_fn handlers for SageMaker inference.
Supports text embedding generation via SageMaker endpoints.
"""

import json
import logging
import os

import torch
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def model_fn(model_dir):
    """Load the SentenceTransformer model from the model directory.

    Args:
        model_dir: Directory where the model artifacts are stored

    Returns:
        Loaded SentenceTransformer model
    """
    logger.info(f"Loading model from {model_dir}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    model = SentenceTransformer(model_dir, device=device)

    # Load training metadata if available
    metadata_path = os.path.join(model_dir, 'training_metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        logger.info(f"Model metadata: {metadata}")

    logger.info("Model loaded successfully")
    return model


def input_fn(request_body, content_type='application/json'):
    """Parse input data for inference.

    Supported formats:
    - "single text"
    - ["text1", "text2"]

    Args:
        request_body: Input request body
        content_type: Content type of the request

    Returns:
        List of input texts
    """
    logger.info(f"Content type: {content_type}")

    if content_type == 'text/plain':
        if not isinstance(request_body, str):
            raise ValueError(f"For text/plain, request body must be a string, got: {type(request_body)}")
        return [request_body]

    if content_type != 'application/json':
        raise ValueError(f"Unsupported content type: {content_type}")

    data = json.loads(request_body)
    if isinstance(data, str):
        return [data]

    if isinstance(data, list):
        if not all(isinstance(item, str) for item in data):
            raise ValueError("Input list must contain only strings")
        return data

    raise ValueError("Unsupported input format. Expected a string or a list of strings.")


def predict_fn(input_data, model):
    """Generate embeddings for input texts.

    Args:
        input_data: List of input texts
        model: Loaded SentenceTransformer model

    Returns:
        Numpy array of embeddings
    """
    logger.info(f"Generating embeddings for {len(input_data)} texts")

    embeddings = model.encode(
        input_data,
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=False
    )

    logger.info(f"Generated embeddings shape: {embeddings.shape}")
    return embeddings


def output_fn(prediction, accept='application/json'):
    """Serialize embeddings to response format.

    Args:
        prediction: Numpy array of embeddings
        accept: Accepted content type

    Returns:
        Serialized response
    """
    if accept == 'application/json':
        # Response format: list of embeddings; each embedding is a list of floats.
        return json.dumps(prediction.tolist())

    else:
        raise ValueError(f"Unsupported accept type: {accept}")
