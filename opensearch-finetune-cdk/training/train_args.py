"""
Training argument parsing for SageMaker
"""
import argparse
import os


def parse_training_args():
    """Parse training arguments from command line and environment variables."""

    parser = argparse.ArgumentParser(description="Fine-tune retrieval models")

    # Model configuration
    parser.add_argument(
        "--model-id",
        type=str,
        required=True,
        help="HuggingFace model ID (e.g., BAAI/bge-base-en-v1.5)"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["dense", "sparse"],
        required=True,
        help="Model type: dense or sparse"
    )

    # Training hyperparameters
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=512,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of training samples (for testing)"
    )
    parser.add_argument(
        "--use-amp",
        action="store_true",
        default=True,
        help="Use automatic mixed precision training"
    )

    # SageMaker directories (set by environment)
    parser.add_argument(
        "--model-dir",
        type=str,
        default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    )
    parser.add_argument(
        "--training-dir",
        type=str,
        default=os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training")
    )
    parser.add_argument(
        "--output-data-dir",
        type=str,
        default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output")
    )

    args = parser.parse_args()
    return args
