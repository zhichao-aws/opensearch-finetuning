"""
Bedrock Batch Orchestrator Lambda Function

Handles synthetic query generation and dev set LLM labeling using Bedrock Batch Inference.
Supports multiple operations: prepare_input, create_job, check_status,
create_dev_label_job, process_dev_labels.

Note: process_output was moved to a SageMaker training job
(training_algo/process_output.py) to avoid Lambda timeout on large datasets.

Operations:
1. prepare_input: Prepare documents for Bedrock batch inference
2. create_job: Create Bedrock batch inference job
3. check_status: Check batch job status
4. create_dev_label_job: Create Bedrock batch job for dev set LLM labeling
5. process_dev_labels: Process dev label output into dev_labels.jsonl

Input varies by operation - see individual function docs.
"""

import io
import json
import os
import random
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

QUERY_GENERATION_PROMPT_WITH_SAMPLES = """You are an expert at generating search queries.

Given the following document, generate 5 diverse search queries that a user might use to find this document.

Here are some example queries from this dataset for reference. Generate queries similar in style to these examples:
{sample_queries}

Document:
{document_text}

Generate exactly 5 queries in the following JSON format:
{{"queries": ["query1", "query2", "query3", "query4", "query5"]}}

Make sure your queries are diverse (different lengths, aspects, intents).
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


def load_sample_queries(s3_uri, max_samples=100):
    """Load sample queries from S3 JSONL file.

    Returns a list of query strings. Up to max_samples are loaded.
    Each line should have 'text', 'query', or 'anchor' field.
    """
    if not s3_uri:
        return []
    try:
        bucket, key = parse_s3_path(s3_uri)
        queries = []
        for record in iter_s3_jsonl(bucket, key, max_lines=max_samples):
            q = record.get('text') or record.get('query') or record.get('anchor') or ''
            if q.strip():
                queries.append(q.strip())
        print(f"Loaded {len(queries)} sample queries from {s3_uri}")
        return queries
    except Exception as e:
        print(f"Warning: failed to load sample queries from {s3_uri}: {e}")
        return []


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
    sample_query_uri = event.get('sample_query_uri', '')
    sample_queries = load_sample_queries(sample_query_uri)

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

            if sample_queries:
                selected = random.sample(sample_queries, min(10, len(sample_queries)))
                prompt = QUERY_GENERATION_PROMPT_WITH_SAMPLES.format(
                    document_text=doc_text,
                    sample_queries='\n'.join(f'- {q}' for q in selected)
                )
            else:
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


def create_dev_label_job(event):
    """
    Create Bedrock batch inference job for dev set LLM relevance labeling.

    The dev_bedrock_input.jsonl was already prepared by process_output.py
    and uploaded to S3.

    Input:
    {
        "operation": "create_dev_label_job",
        "dev_bedrock_s3_path": "s3://bucket/dev-labeling/input/dev_bedrock_input.jsonl"
    }

    Output:
    {
        "job_arn": "arn:aws:bedrock:...",
        "job_id": "dev-label-..."
    }
    """
    dev_bedrock_s3_path = event.get('dev_bedrock_s3_path')
    if not dev_bedrock_s3_path:
        raise ValueError("dev_bedrock_s3_path is required")

    bucket = os.environ.get('DATA_BUCKET')
    model_id = os.environ.get('DEV_LABEL_MODEL_ID') or os.environ.get('BEDROCK_MODEL_ID', 'us.anthropic.claude-sonnet-4-6-v1:0')
    role_arn = os.environ.get('BEDROCK_BATCH_ROLE_ARN')

    if not role_arn:
        raise ValueError("BEDROCK_BATCH_ROLE_ARN environment variable is required")

    model_name = event.get('model_name', '')
    prefix = f"{model_name}/" if model_name else ""

    timestamp = datetime.utcnow().strftime('%Y%m%d-%H%M%S')
    output_prefix = f"{prefix}dev-labeling/output/{timestamp}/"

    bedrock = boto3.client('bedrock')

    job_name = f"dev-label-{timestamp}"

    response = bedrock.create_model_invocation_job(
        jobName=job_name,
        modelId=model_id,
        roleArn=role_arn,
        inputDataConfig={
            "s3InputDataConfig": {
                "s3Uri": dev_bedrock_s3_path
            }
        },
        outputDataConfig={
            "s3OutputDataConfig": {
                "s3Uri": f"s3://{bucket}/{output_prefix}"
            }
        }
    )

    job_arn = response['jobArn']
    print(f"Created dev label batch job: {job_arn}")

    return {
        "statusCode": 200,
        "job_arn": job_arn,
        "job_id": job_name
    }


def process_dev_labels(event):
    """
    Process Bedrock batch output for dev set LLM labels.

    Parses the output, extracts relevance scores, and writes dev_labels.jsonl
    alongside the dev_data.jsonl in S3.

    Input:
    {
        "operation": "process_dev_labels",
        "output_s3_path": "s3://bucket/dev-labeling/output/...",
        "dev_data_s3_path": "s3://bucket/dev-labeling/dev_data.jsonl"
    }

    Output:
    {
        "dev_labels_s3_path": "s3://bucket/dev-labeling/dev_labels.jsonl",
        "num_labels": 50000
    }
    """
    output_s3_path = event.get('output_s3_path')
    dev_data_s3_path = event.get('dev_data_s3_path')

    if not output_s3_path:
        raise ValueError("output_s3_path is required")

    bucket = os.environ.get('DATA_BUCKET')
    s3 = boto3.client('s3')

    # Load dev_data to reconstruct (query, candidate_id) mapping from record IDs
    dev_samples = []
    if dev_data_s3_path:
        dev_bucket, dev_key = parse_s3_path(dev_data_s3_path)
        for record in iter_s3_jsonl(dev_bucket, dev_key):
            dev_samples.append(record)

    # Build record_id -> (query, candidate_id) mapping
    record_to_qd = {}
    record_idx = 0
    for sample in dev_samples:
        query = sample["query"]
        for cand in sample["candidates"]:
            record_id = f"dev_{record_idx}"
            record_to_qd[record_id] = (query, cand["id"])
            record_idx += 1

    print(f"Built mapping for {len(record_to_qd)} records from {len(dev_samples)} dev queries")

    # List output files
    output_bucket, output_prefix = parse_s3_path(output_s3_path)
    paginator = s3.get_paginator('list_objects_v2')
    output_files = []
    for page in paginator.paginate(Bucket=output_bucket, Prefix=output_prefix):
        for obj in page.get('Contents', []):
            if obj['Key'].endswith('.jsonl.out'):
                output_files.append(obj['Key'])

    print(f"Found {len(output_files)} dev label output files")

    # Parse output and extract scores
    labels = []
    errors = 0

    for output_key in output_files:
        try:
            response = s3.get_object(Bucket=output_bucket, Key=output_key)
            content = response['Body'].read().decode('utf-8')

            for line in content.strip().split('\n'):
                if not line.strip():
                    continue

                try:
                    record = json.loads(line)
                    record_id = record.get('recordId')
                    model_output = record.get('modelOutput', {})

                    if record_id not in record_to_qd:
                        continue

                    query, candidate_id = record_to_qd[record_id]

                    # Extract score from model output
                    score = _extract_score_from_output(model_output, candidate_id)

                    labels.append({
                        "query": query,
                        "candidate_id": candidate_id,
                        "llm_score": score
                    })

                except Exception as e:
                    print(f"Error processing dev label record: {e}")
                    errors += 1

        except Exception as e:
            print(f"Error processing dev label file {output_key}: {e}")

    print(f"Extracted {len(labels)} labels, {errors} errors")

    # Write dev_labels.jsonl to S3 (namespaced by model_name)
    model_name = event.get('model_name', '')
    prefix = f"{model_name}/" if model_name else ""
    labels_content = '\n'.join(json.dumps(l) for l in labels) + '\n'
    dev_labels_s3_path = write_s3_file(
        bucket,
        f"{prefix}dev-labeling/dev_labels.jsonl",
        labels_content,
        'application/jsonl'
    )

    print(f"Dev labels written to {dev_labels_s3_path}")

    return {
        "statusCode": 200,
        "dev_labels_s3_path": dev_labels_s3_path,
        "num_labels": len(labels)
    }


def _extract_score_from_output(model_output, doc_id):
    """Extract relevance score from Claude model output for dev labeling."""
    import re

    output_content = model_output.get('content', [])
    response_text = ''
    for block in output_content:
        if block.get('type') == 'text':
            response_text = block.get('text', '')
            break

    if not response_text:
        return 0.0

    # Clean markdown
    parsed_text = response_text.strip()
    if '```json' in parsed_text:
        parsed_text = parsed_text.split('```json')[1].split('```')[0].strip()
    elif '```' in parsed_text:
        parsed_text = parsed_text.split('```')[1].split('```')[0].strip()

    # Try JSON parsing
    try:
        result = json.loads(parsed_text)
        ratings = result.get('ratings', [])
        if ratings:
            score = ratings[0].get('rating_score')
            if score is not None and 0 <= float(score) <= 1:
                return float(score)
    except (json.JSONDecodeError, ValueError):
        pass

    # Try fixing common JSON issues
    try:
        fixed_text = re.sub(r',\s*([}\]])', r'\1', parsed_text)
        result = json.loads(fixed_text)
        ratings = result.get('ratings', [])
        if ratings:
            score = ratings[0].get('rating_score')
            if score is not None and 0 <= float(score) <= 1:
                return float(score)
    except (json.JSONDecodeError, ValueError):
        pass

    # Regex fallback
    patterns = [
        r'"rating_score"\s*:\s*([0-9]*\.?[0-9]+)',
        r'"score"\s*:\s*([0-9]*\.?[0-9]+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, response_text)
        if match:
            try:
                score = float(match.group(1))
                if 0 <= score <= 1:
                    return score
            except ValueError:
                continue

    return 0.0


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
    elif operation == 'create_dev_label_job':
        return create_dev_label_job(event)
    elif operation == 'process_dev_labels':
        return process_dev_labels(event)
    else:
        raise ValueError(f"Unknown operation: {operation}. Valid operations: prepare_input, create_job, check_status, create_dev_label_job, process_dev_labels")
