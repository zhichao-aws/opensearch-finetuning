"""
Bedrock Batch Orchestrator Lambda Function

Handles synthetic query generation using Bedrock Batch Inference.
Supports multiple operations: prepare_input, create_job, check_status.

Note: process_output was moved to a SageMaker training job
(training_algo/process_output.py) to avoid Lambda timeout on large datasets.

Operations:
1. prepare_input: Prepare documents for Bedrock batch inference
2. create_job: Create Bedrock batch inference job
3. check_status: Check batch job status

Input varies by operation - see individual function docs.
"""

import io
import json
import os
from datetime import datetime
from urllib.parse import urlparse

import boto3


# =============================================================================
# Prompt template for query generation
# =============================================================================

QUERY_GENERATION_PROMPT = """You are an expert at generating search queries.

Given the following document, generate 5 diverse search queries that a user might use to find this document.

Document:
{document_text}

Generate exactly 5 queries in the following JSON format:
{{"queries": ["query1", "query2", "query3", "query4", "query5"]}}

Only return the JSON, no additional text."""


def parse_s3_path(s3_path):
    """Parse S3 path into bucket and key."""
    parsed = urlparse(s3_path)
    bucket = parsed.netloc
    key = parsed.path.lstrip('/')
    return bucket, key


def iter_s3_jsonl(bucket, key, max_lines=None):
    """Stream JSONL file from S3 line by line, yielding parsed JSON objects."""
    s3 = boto3.client('s3')
    response = s3.get_object(Bucket=bucket, Key=key)
    stream = io.TextIOWrapper(response['Body'], encoding='utf-8')

    count = 0
    for line in stream:
        line = line.strip()
        if not line:
            continue
        yield json.loads(line)
        count += 1
        if max_lines and count >= max_lines:
            break


def write_s3_file(bucket, key, content, content_type='application/json'):
    """Write content to S3."""
    s3 = boto3.client('s3')
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=content.encode('utf-8') if isinstance(content, str) else content,
        ContentType=content_type
    )
    return f"s3://{bucket}/{key}"


def prepare_input(event):
    """
    Prepare documents for Bedrock batch inference.
    Uses S3 multipart upload to stream output without accumulating in memory.

    Input:
    {
        "operation": "prepare_input",
        "s3_documents_path": "s3://bucket/raw-corpus/documents.jsonl",
        "max_documents": 20000  # Optional, limits number of documents to process
    }

    Output:
    {
        "s3_input_path": "s3://bucket/bedrock-batch/input/batch_input.jsonl",
        "record_count": 20000
    }
    """
    s3_documents_path = event.get('s3_documents_path')
    if not s3_documents_path:
        raise ValueError("s3_documents_path is required")

    bucket = os.environ.get('DATA_BUCKET')
    max_documents = event.get('max_documents')

    # Stream documents from S3 (only read up to max_documents)
    input_bucket, input_key = parse_s3_path(s3_documents_path)

    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    s3_input_key = f"bedrock-batch/input/batch_input_{timestamp}.jsonl"

    # Use multipart upload to stream output to S3 without holding everything in memory
    s3 = boto3.client('s3')
    mpu = s3.create_multipart_upload(Bucket=bucket, Key=s3_input_key, ContentType='application/jsonl')
    upload_id = mpu['UploadId']

    parts = []
    part_number = 1
    buffer = io.BytesIO()
    # S3 multipart minimum part size is 5MB (except last part)
    PART_SIZE = 6 * 1024 * 1024
    record_count = 0

    try:
        for doc_idx, doc in enumerate(iter_s3_jsonl(input_bucket, input_key, max_lines=max_documents)):
            doc_id = doc.get('id', str(doc_idx))
            doc_text = doc.get('text', '')

            # Truncate document if too long (Bedrock has input limits)
            max_doc_length = 8000  # Leave room for prompt template
            if len(doc_text) > max_doc_length:
                doc_text = doc_text[:max_doc_length] + "..."

            prompt = QUERY_GENERATION_PROMPT.format(document_text=doc_text)

            record = {
                "recordId": doc_id,
                "modelInput": {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 1024,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                }
            }

            line = json.dumps(record) + '\n'
            buffer.write(line.encode('utf-8'))
            record_count += 1

            # Flush buffer as a part when it exceeds the threshold
            if buffer.tell() >= PART_SIZE:
                buffer.seek(0)
                part = s3.upload_part(
                    Bucket=bucket, Key=s3_input_key,
                    UploadId=upload_id, PartNumber=part_number,
                    Body=buffer.read()
                )
                parts.append({'ETag': part['ETag'], 'PartNumber': part_number})
                part_number += 1
                buffer = io.BytesIO()

        # Upload remaining data
        if buffer.tell() > 0:
            buffer.seek(0)
            part = s3.upload_part(
                Bucket=bucket, Key=s3_input_key,
                UploadId=upload_id, PartNumber=part_number,
                Body=buffer.read()
            )
            parts.append({'ETag': part['ETag'], 'PartNumber': part_number})

        s3.complete_multipart_upload(
            Bucket=bucket, Key=s3_input_key,
            UploadId=upload_id,
            MultipartUpload={'Parts': parts}
        )
    except Exception:
        s3.abort_multipart_upload(Bucket=bucket, Key=s3_input_key, UploadId=upload_id)
        raise

    s3_input_path = f"s3://{bucket}/{s3_input_key}"
    print(f"Batch input written to {s3_input_path}")

    return {
        "statusCode": 200,
        "s3_input_path": s3_input_path,
        "record_count": record_count
    }


def create_job(event):
    """
    Create Bedrock batch inference job.

    Input:
    {
        "operation": "create_job",
        "s3_input_path": "s3://bucket/bedrock-batch/input/batch_input.jsonl"
    }

    Output:
    {
        "job_arn": "arn:aws:bedrock:...",
        "job_id": "job-id"
    }
    """
    s3_input_path = event.get('s3_input_path')
    if not s3_input_path:
        raise ValueError("s3_input_path is required")

    bucket = os.environ.get('DATA_BUCKET')
    model_id = os.environ.get('BEDROCK_MODEL_ID', 'us.anthropic.claude-haiku-4-5-20251001-v1:0')
    role_arn = os.environ.get('BEDROCK_BATCH_ROLE_ARN')

    if not role_arn:
        raise ValueError("BEDROCK_BATCH_ROLE_ARN environment variable is required")

    # Parse input path
    input_bucket, input_key = parse_s3_path(s3_input_path)

    # Create output path
    timestamp = datetime.utcnow().strftime('%Y%m%d-%H%M%S')
    output_prefix = f"bedrock-batch/output/{timestamp}/"

    bedrock = boto3.client('bedrock')

    # Create batch inference job
    job_name = f"query-gen-{timestamp}"

    response = bedrock.create_model_invocation_job(
        jobName=job_name,
        modelId=model_id,
        roleArn=role_arn,
        inputDataConfig={
            "s3InputDataConfig": {
                "s3Uri": s3_input_path
            }
        },
        outputDataConfig={
            "s3OutputDataConfig": {
                "s3Uri": f"s3://{bucket}/{output_prefix}"
            }
        }
    )

    job_arn = response['jobArn']
    print(f"Created batch job: {job_arn}")

    return {
        "statusCode": 200,
        "job_arn": job_arn,
        "job_id": job_name
    }


def check_status(event):
    """
    Check Bedrock batch job status.

    Input:
    {
        "operation": "check_status",
        "job_arn": "arn:aws:bedrock:..."
    }

    Output:
    {
        "status": "Completed|InProgress|Failed|Stopped",
        "output_s3_path": "s3://bucket/bedrock-batch/output/...",  # if completed
        "message": "status message"
    }
    """
    job_arn = event.get('job_arn')
    if not job_arn:
        raise ValueError("job_arn is required")

    bedrock = boto3.client('bedrock')

    response = bedrock.get_model_invocation_job(jobIdentifier=job_arn)

    status = response['status']
    message = response.get('message', '')

    # Always include output_s3_path (empty string if not completed) so Step Functions ResultSelector doesn't fail
    output_s3_path = ""
    if status == 'Completed':
        output_config = response.get('outputDataConfig', {})
        s3_config = output_config.get('s3OutputDataConfig', {})
        output_s3_path = s3_config.get('s3Uri', '')

    result = {
        "statusCode": 200,
        "status": status,
        "message": message,
        "output_s3_path": output_s3_path
    }

    print(f"Job status: {status}, message: {message}")

    return result


def handler(event, context):
    """Lambda handler - routes to appropriate operation."""
    print(f"Event: {json.dumps(event)}")

    operation = event.get('operation')

    if operation == 'prepare_input':
        return prepare_input(event)
    elif operation == 'create_job':
        return create_job(event)
    elif operation == 'check_status':
        return check_status(event)
    else:
        raise ValueError(f"Unknown operation: {operation}. Valid operations: prepare_input, create_job, check_status")
