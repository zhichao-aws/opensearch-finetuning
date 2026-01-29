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
    Validate S3 corpus format and process it.

    Expected format: JSONL with {"id": "...", "text": "..."}
    or txt file with one document per line.

    Args:
        s3_corpus_path: S3 path to corpus file
        max_documents: Maximum number of documents to process

    Returns:
        Dictionary with validation result
    """
    logger.info(f"Validating corpus at: {s3_corpus_path}")

    try:
        # Parse S3 path
        source_bucket, source_key = parse_s3_path(s3_corpus_path)

        # Download file
        logger.info(f"Downloading from s3://{source_bucket}/{source_key}")
        response = s3_client.get_object(Bucket=source_bucket, Key=source_key)
        content = response["Body"].read().decode("utf-8")

        # Parse documents
        documents = []
        lines = content.strip().split("\n")

        logger.info(f"Processing {len(lines)} lines")

        for idx, line in enumerate(lines[:max_documents]):
            line = line.strip()
            if not line:
                continue

            # Try to parse as JSON
            try:
                doc = json.loads(line)
                if "text" in doc:
                    # Already in correct format
                    if "id" not in doc:
                        doc["id"] = str(idx)
                    documents.append(doc)
                else:
                    # Plain text, wrap in format
                    documents.append({
                        "id": str(idx),
                        "text": line
                    })
            except json.JSONDecodeError:
                # Plain text file
                documents.append({
                    "id": str(idx),
                    "text": line
                })

        logger.info(f"Validated {len(documents)} documents")

        # Save to standard location
        dest_key = "raw-corpus/documents.jsonl"
        jsonl_content = "\n".join(json.dumps(doc) for doc in documents)

        s3_client.put_object(
            Bucket=DATA_BUCKET,
            Key=dest_key,
            Body=jsonl_content.encode("utf-8"),
            ContentType="application/jsonlines"
        )

        dest_path = f"s3://{DATA_BUCKET}/{dest_key}"
        logger.info(f"Saved processed corpus to {dest_path}")

        return {
            "status": "success",
            "s3_path": dest_path,
            "document_count": len(documents),
            "data_bucket": DATA_BUCKET,
        }

    except Exception as e:
        logger.error(f"Error validating corpus: {str(e)}", exc_info=True)
        raise


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
