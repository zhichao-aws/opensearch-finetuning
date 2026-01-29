"""
Training constants
"""

# SageMaker paths
DEFAULT_MODEL_DIR = "/opt/ml/model"
DEFAULT_TRAINING_DIR = "/opt/ml/input/data/training"
DEFAULT_OUTPUT_DIR = "/opt/ml/output"

# Training defaults
DEFAULT_NUM_EPOCHS = 3
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 2e-5
DEFAULT_MAX_SEQ_LENGTH = 512

# Model types
MODEL_TYPE_DENSE = "dense"
MODEL_TYPE_SPARSE = "sparse"
