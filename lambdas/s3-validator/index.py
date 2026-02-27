"""
S3 Data Validator Lambda Function

Validates that S3 corpus is in JSONL format with required 'text' field.
Raises an error if validation fails, causing the state machine execution to fail.

Input:
{
    "s3_corpus_path": "s3://bucket/path/corpus.jsonl"
}

Output (on success):
{
    "s3_corpus_path": "s3://bucket/path/corpus.jsonl",
    "document_count": 1000
}
"""

import json
import boto3
from urllib.parse import urlparse


def parse_s3_path(s3_path):
    """Parse S3 path into bucket and key."""
    parsed = urlparse(s3_path)
    bucket = parsed.netloc
    key = parsed.path.lstrip('/')
    return bucket, key


def handler(event, context):
    """Lambda handler for S3 corpus validation."""
    print(f"Event: {json.dumps(event)}")

    s3_corpus_path = event.get('s3_corpus_path')
    if not s3_corpus_path:
        raise ValueError("s3_corpus_path is required")

    bucket, key = parse_s3_path(s3_corpus_path)
    print(f"Validating s3://{bucket}/{key}")

    # Check file extension
    if not key.endswith('.jsonl'):
        raise ValueError(f"File must be JSONL format (.jsonl extension), got: {key}")

    # Read and validate content
    s3 = boto3.client('s3')
    response = s3.get_object(Bucket=bucket, Key=key)
    content = response['Body'].read().decode('utf-8')

    document_count = 0
    for line_num, line in enumerate(content.strip().split('\n'), start=1):
        line = line.strip()
        if not line:
            continue

        # Validate JSON format
        try:
            doc = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"Line {line_num} is not valid JSON: {e}")

        # Validate 'text' field exists
        if 'text' not in doc:
            raise ValueError(f"Line {line_num} missing required 'text' field")

        document_count += 1

    if document_count == 0:
        raise ValueError("File contains no valid documents")

    print(f"Validation passed: {document_count} documents")

    return {
        "s3_corpus_path": s3_corpus_path,
        "document_count": document_count
    }
