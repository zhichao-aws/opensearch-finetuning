"""
Lambda Function: Bedrock Batch Orchestrator
Orchestrates Bedrock Batch Inference for synthetic query generation
"""
import json
import logging
import os
from datetime import datetime, timezone
from typing import Dict, List

import boto3
from prompt_templates import format_prompt_for_document

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3_client = boto3.client("s3")
bedrock_client = boto3.client("bedrock")
DATA_BUCKET = os.environ["DATA_BUCKET"]
BEDROCK_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0")


def parse_s3_path(s3_path: str) -> tuple:
    """Parse S3 path into bucket and key."""
    if s3_path.startswith("s3://"):
        s3_path = s3_path[5:]
    parts = s3_path.split("/", 1)
    return parts[0], parts[1] if len(parts) > 1 else ""


def prepare_bedrock_input(s3_documents_path: str) -> Dict:
    """
    Prepare Bedrock Batch input from documents.

    Args:
        s3_documents_path: S3 path to documents JSONL

    Returns:
        Dictionary with input file S3 path
    """
    logger.info(f"Preparing Bedrock input from: {s3_documents_path}")

    # Parse S3 path
    bucket, key = parse_s3_path(s3_documents_path)

    # Download documents
    response = s3_client.get_object(Bucket=bucket, Key=key)
    content = response["Body"].read().decode("utf-8")

    documents = []
    for line in content.strip().split("\n"):
        if line:
            documents.append(json.loads(line))

    logger.info(f"Loaded {len(documents)} documents")

    # Convert to Bedrock Batch format
    bedrock_inputs = []
    for doc in documents:
        prompt = format_prompt_for_document(doc["text"])

        bedrock_input = {
            "recordId": doc["id"],
            "modelInput": {
                "anthropic_version": "bedrock-2023-05-31",
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": 500,
                "temperature": 0.7
            }
        }
        bedrock_inputs.append(bedrock_input)

    # Save to S3 with timestamp to avoid conflicts
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    bedrock_input_key = f"bedrock-input/batch_input_{timestamp}.jsonl"
    bedrock_input_content = "\n".join(json.dumps(inp) for inp in bedrock_inputs)

    s3_client.put_object(
        Bucket=DATA_BUCKET,
        Key=bedrock_input_key,
        Body=bedrock_input_content.encode("utf-8"),
        ContentType="application/jsonlines"
    )

    s3_input_path = f"s3://{DATA_BUCKET}/{bedrock_input_key}"
    logger.info(f"Bedrock input saved to {s3_input_path}")

    return {
        "status": "success",
        "s3_input_path": s3_input_path,
        "record_count": len(bedrock_inputs)
    }


def create_bedrock_batch_job(s3_input_path: str, bedrock_batch_role_arn: str) -> Dict:
    """
    Create Bedrock Batch Inference job.

    Args:
        s3_input_path: S3 path to input JSONL
        bedrock_batch_role_arn: IAM role ARN for Bedrock

    Returns:
        Dictionary with job ID and status
    """
    logger.info(f"Creating Bedrock Batch job for: {s3_input_path}")

    # Parse input path
    input_bucket, input_key = parse_s3_path(s3_input_path)

    # Output path
    output_s3_uri = f"s3://{DATA_BUCKET}/bedrock-output/"

    try:
        # Create batch job
        response = bedrock_client.create_model_invocation_job(
            jobName=f"query-generation-{os.urandom(4).hex()}",
            modelId=BEDROCK_MODEL_ID,
            roleArn=bedrock_batch_role_arn,
            inputDataConfig={
                "s3InputDataConfig": {
                    "s3Uri": s3_input_path
                }
            },
            outputDataConfig={
                "s3OutputDataConfig": {
                    "s3Uri": output_s3_uri
                }
            }
        )

        job_arn = response["jobArn"]
        # Extract job ID from ARN
        job_id = job_arn.split("/")[-1]

        logger.info(f"Created Bedrock Batch job: {job_id}")

        return {
            "status": "success",
            "job_id": job_id,
            "job_arn": job_arn,
            "output_s3_uri": output_s3_uri
        }

    except Exception as e:
        logger.error(f"Error creating Bedrock Batch job: {str(e)}", exc_info=True)
        raise


def check_bedrock_job_status(job_id: str) -> Dict:
    """
    Check Bedrock Batch job status.

    Args:
        job_id: Bedrock job ID

    Returns:
        Dictionary with job status
    """
    logger.info(f"Checking status for job: {job_id}")

    try:
        response = bedrock_client.get_model_invocation_job(jobIdentifier=job_id)

        status = response["status"]
        logger.info(f"Job {job_id} status: {status}")

        result = {
            "status": status,
            "job_id": job_id
        }

        # Add output path if completed
        if status == "Completed":
            output_config = response.get("outputDataConfig", {})
            output_s3_config = output_config.get("s3OutputDataConfig", {})
            result["output_s3_path"] = output_s3_config.get("s3Uri", "")

        return result

    except Exception as e:
        logger.error(f"Error checking job status: {str(e)}", exc_info=True)
        raise


def process_bedrock_output(output_s3_path: str, s3_documents_path: str) -> Dict:
    """
    Process Bedrock Batch output and create training data.

    Args:
        output_s3_path: S3 path to Bedrock output
        s3_documents_path: S3 path to original documents

    Returns:
        Dictionary with training data S3 path
    """
    logger.info(f"Processing Bedrock output from: {output_s3_path}")

    # Parse S3 path
    bucket, prefix = parse_s3_path(output_s3_path)

    # List files in output directory
    response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
    output_files = [obj["Key"] for obj in response.get("Contents", []) if obj["Key"].endswith(".jsonl.out")]

    if not output_files:
        raise ValueError(f"No output files found in {output_s3_path}")

    logger.info(f"Found {len(output_files)} output files")

    # Download and parse first output file
    output_key = output_files[0]
    response = s3_client.get_object(Bucket=bucket, Key=output_key)
    output_content = response["Body"].read().decode("utf-8")

    # Also load original documents for pairing
    doc_bucket, doc_key = parse_s3_path(s3_documents_path)
    doc_response = s3_client.get_object(Bucket=doc_bucket, Key=doc_key)
    doc_content = doc_response["Body"].read().decode("utf-8")

    documents_by_id = {}
    for line in doc_content.strip().split("\n"):
        if line:
            doc = json.loads(line)
            documents_by_id[doc["id"]] = doc["text"]

    # Process Bedrock outputs
    training_pairs = []
    for line in output_content.strip().split("\n"):
        if not line:
            continue

        try:
            result = json.loads(line)
            record_id = result["recordId"]
            model_output = result.get("modelOutput", {})

            # Extract queries from model response
            content = model_output.get("content", [])
            if content and len(content) > 0:
                text_content = content[0].get("text", "")

                # Parse JSON response
                try:
                    queries_data = json.loads(text_content)
                    queries = queries_data.get("queries", [])

                    # Create training pairs
                    document_text = documents_by_id.get(record_id, "")
                    for query in queries:
                        training_pairs.append({
                            "query": query,
                            "document": document_text
                        })
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse queries for record {record_id}")
                    continue

        except Exception as e:
            logger.warning(f"Error processing output line: {str(e)}")
            continue

    logger.info(f"Created {len(training_pairs)} training pairs")

    # Save training data with timestamp to avoid conflicts
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    training_data_key = f"training-data/training_data_{timestamp}.jsonl"
    training_data_content = "\n".join(json.dumps(pair) for pair in training_pairs)

    s3_client.put_object(
        Bucket=DATA_BUCKET,
        Key=training_data_key,
        Body=training_data_content.encode("utf-8"),
        ContentType="application/jsonlines"
    )

    training_data_s3_path = f"s3://{DATA_BUCKET}/{training_data_key}"
    logger.info(f"Training data saved to {training_data_s3_path}")

    return {
        "status": "success",
        "training_data_s3_path": training_data_s3_path,
        "training_pair_count": len(training_pairs)
    }


def handler(event, context):
    """Lambda handler for Bedrock Batch orchestration."""
    logger.info(f"Event: {json.dumps(event)}")

    try:
        operation = event["operation"]

        if operation == "prepare_input":
            s3_documents_path = event["s3_documents_path"]
            result = prepare_bedrock_input(s3_documents_path)

        elif operation == "create_job":
            s3_input_path = event["s3_input_path"]
            bedrock_batch_role_arn = event["bedrock_batch_role_arn"]
            result = create_bedrock_batch_job(s3_input_path, bedrock_batch_role_arn)

        elif operation == "check_status":
            job_id = event["job_id"]
            result = check_bedrock_job_status(job_id)

        elif operation == "process_output":
            output_s3_path = event["output_s3_path"]
            s3_documents_path = event["s3_documents_path"]
            result = process_bedrock_output(output_s3_path, s3_documents_path)

        else:
            raise ValueError(f"Unknown operation: {operation}")

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
