"""
Main training script for fine-tuning retrieval models
Uses sentence-transformers with contrastive learning
"""
import json
import logging
import os
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer, losses
from torch.utils.data import DataLoader

from train_args import parse_training_args
from data_loader import load_training_data
from model_utils import initialize_model

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main training function compatible with SageMaker."""

    # Parse arguments
    args = parse_training_args()

    # SageMaker environment variables
    model_dir = args.model_dir
    training_dir = args.training_dir
    output_data_dir = args.output_data_dir

    logger.info("=" * 80)
    logger.info("Starting Fine-Tuning Job")
    logger.info("=" * 80)
    logger.info(f"Model ID: {args.model_id}")
    logger.info(f"Model Type: {args.model_type}")
    logger.info(f"Epochs: {args.num_epochs}")
    logger.info(f"Batch Size: {args.batch_size}")
    logger.info(f"Learning Rate: {args.learning_rate}")
    logger.info(f"Max Seq Length: {args.max_seq_length}")
    logger.info(f"Training Dir: {training_dir}")
    logger.info(f"Model Dir: {model_dir}")
    logger.info("=" * 80)

    # Check GPU availability
    if torch.cuda.is_available():
        logger.info(f"GPU Available: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        logger.warning("No GPU available, training on CPU")

    # Load training data from S3
    logger.info("Loading training data...")
    train_examples = load_training_data(training_dir, max_samples=args.max_samples)
    logger.info(f"Loaded {len(train_examples)} training examples")

    if len(train_examples) == 0:
        raise ValueError("No training examples found!")

    # Initialize model
    logger.info(f"Initializing model: {args.model_id}")
    model = initialize_model(
        model_id=args.model_id,
        model_type=args.model_type,
        max_seq_length=args.max_seq_length
    )

    # Move model to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    logger.info(f"Model loaded on device: {device}")

    # Create DataLoader
    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=args.batch_size
    )
    logger.info(f"Created DataLoader with batch size {args.batch_size}")
    logger.info(f"Total batches per epoch: {len(train_dataloader)}")

    # Define loss function
    # MultipleNegativesRankingLoss: Uses in-batch negatives for contrastive learning
    # Given pairs (query, doc), treats other docs in batch as negatives
    if args.model_type == "dense":
        train_loss = losses.MultipleNegativesRankingLoss(model)
        logger.info("Using MultipleNegativesRankingLoss for dense model")
    else:  # sparse
        train_loss = losses.MultipleNegativesRankingLoss(model)
        logger.info("Using MultipleNegativesRankingLoss for sparse model")

    # Training configuration
    num_epochs = args.num_epochs
    warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)  # 10% warmup
    total_steps = len(train_dataloader) * num_epochs

    logger.info("=" * 80)
    logger.info("Training Configuration")
    logger.info("=" * 80)
    logger.info(f"Epochs: {num_epochs}")
    logger.info(f"Steps per epoch: {len(train_dataloader)}")
    logger.info(f"Total steps: {total_steps}")
    logger.info(f"Warmup steps: {warmup_steps}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Use AMP: {args.use_amp}")
    logger.info("=" * 80)

    # Train the model
    logger.info("Starting training...")

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        optimizer_params={'lr': args.learning_rate},
        checkpoint_path=output_data_dir,
        checkpoint_save_steps=1000,
        show_progress_bar=True,
        use_amp=args.use_amp  # Automatic Mixed Precision
    )

    logger.info("Training completed successfully!")

    # Save model to SageMaker model directory
    logger.info(f"Saving model to {model_dir}")
    model.save(model_dir)
    logger.info("Model saved successfully")

    # Save training metadata
    metadata = {
        "model_type": args.model_type,
        "base_model_id": args.model_id,
        "num_epochs": num_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "training_samples": len(train_examples),
        "max_seq_length": args.max_seq_length,
        "total_training_steps": total_steps,
        "warmup_steps": warmup_steps,
        "use_amp": args.use_amp,
    }

    metadata_path = os.path.join(model_dir, "training_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Training metadata saved to {metadata_path}")

    logger.info("=" * 80)
    logger.info("Training job completed successfully!")
    logger.info("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}", exc_info=True)
        raise
