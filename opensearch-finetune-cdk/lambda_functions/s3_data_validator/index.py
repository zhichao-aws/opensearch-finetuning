"""
Lambda Function: S3 Data Validator
Validates S3 corpus format and copies to standard location
"""
import json
import logging
import os
from typing import Dict, List

import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3_client = boto3.client("s3")
DATA_BUCKET = os.environ["DATA_BUCKET"]
MAX_DOCUMENTS = int(os.environ.get("MAX_DOCUMENTS", "5000"))


def parse_s3_path(s3_path: str) -> tuple:
    """Parse S3 path into bucket and key."""
    if s3_path.startswith("s3://"):
        s3_path = s3_path[5:]

    parts = s3_path.split("/", 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ""

    return bucket, key


def validate_and_process_corpus(s3_corpus_path: str, max_documents: int) -> Dict:
    """
    Validate S3 corpus format.

    Expected format: JSONL with {"id": "...", "text": "..."} on each line.
    Both 'id' and 'text' fields are required.

    Args:
        s3_corpus_path: S3 path to corpus file (must be valid JSONL)
        max_documents: Maximum number of documents to validate

    Returns:
        Dictionary with validation result

    Raises:
        ValueError: If format is invalid or required fields are missing
    """
    logger.info(f"Validating corpus at: {s3_corpus_path}")

    try:
        # Parse S3 path
        source_bucket, source_key = parse_s3_path(s3_corpus_path)

        # Check file extension
        if not source_key.endswith('.jsonl'):
            raise ValueError(
                f"Invalid file format. Expected .jsonl file, got: {source_key}. "
                "Only JSONL format is supported."
            )

        # Download file
        logger.info(f"Downloading from s3://{source_bucket}/{source_key}")
        response = s3_client.get_object(Bucket=source_bucket, Key=source_key)
        content = response["Body"].read().decode("utf-8")

        # Parse and validate documents
        lines = content.strip().split("\n")
        logger.info(f"Validating {min(len(lines), max_documents)} lines")

        document_count = 0
        for line_num, line in enumerate(lines[:max_documents], start=1):
            line = line.strip()
            if not line:
                continue

            # Parse as JSON
            try:
                doc = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON on line {line_num}: {str(e)}. "
                    "Each line must be valid JSON."
                )

            # Validate required fields
            if "id" not in doc:
                raise ValueError(
                    f"Missing required field 'id' on line {line_num}. "
                    "Each document must have 'id' and 'text' fields."
                )

            if "text" not in doc:
                raise ValueError(
                    f"Missing required field 'text' on line {line_num}. "
                    "Each document must have 'id' and 'text' fields."
                )

            # Validate field types
            if not isinstance(doc["id"], str):
                raise ValueError(
                    f"Field 'id' must be a string on line {line_num}, "
                    f"got {type(doc['id']).__name__}"
                )

            if not isinstance(doc["text"], str):
                raise ValueError(
                    f"Field 'text' must be a string on line {line_num}, "
                    f"got {type(doc['text']).__name__}"
                )

            # Validate non-empty
            if not doc["text"].strip():
                raise ValueError(
                    f"Field 'text' cannot be empty on line {line_num}"
                )

            document_count += 1

        logger.info(f"Validated {document_count} documents successfully")

        return {
            "status": "success",
            "s3_path": s3_corpus_path,  # Return original path
            "document_count": document_count,
        }

    except ValueError:
        # Re-raise validation errors
        raise
    except Exception as e:
        logger.error(f"Error validating corpus: {str(e)}", exc_info=True)
        raise ValueError(f"Error reading or validating corpus: {str(e)}")


def handler(event, context):
    """Lambda handler for S3 data validation."""
    logger.info(f"Event: {json.dumps(event)}")

    try:
        # Extract parameters
        s3_corpus_path = event["s3_corpus_path"]
        max_documents = int(event.get("max_documents", MAX_DOCUMENTS))

        # Validate and process corpus
        result = validate_and_process_corpus(s3_corpus_path, max_documents)

        return {
            "statusCode": 200,
            **result
        }

    except Exception as e:
        logger.error(f"Error in handler: {str(e)}", exc_info=True)
        return {
            "statusCode": 500,
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__,
        }
