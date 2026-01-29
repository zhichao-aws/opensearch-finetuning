"""
Lambda Function: Data Extractor
Extracts documents from OpenSearch using PIT/Scroll API and saves to S3
"""
import json
import logging
import os
import random
from typing import Dict, List

import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3_client = boto3.client("s3")
DATA_BUCKET = os.environ["DATA_BUCKET"]
MAX_DOCUMENTS = int(os.environ.get("MAX_DOCUMENTS", "5000"))


def get_opensearch_client(endpoint: str) -> OpenSearch:
    """Create OpenSearch client with IAM authentication."""
    # Remove https:// prefix if present
    host = endpoint.replace("https://", "").replace("http://", "")

    # Get AWS credentials
    credentials = boto3.Session().get_credentials()
    awsauth = AWS4Auth(
        credentials.access_key,
        credentials.secret_key,
        boto3.Session().region_name or "us-east-1",
        "es",
        session_token=credentials.token,
    )

    client = OpenSearch(
        hosts=[{"host": host, "port": 443}],
        http_auth=awsauth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        timeout=30,
    )

    return client


def extract_documents_from_opensearch(
    endpoint: str,
    index_name: str,
    text_field: str = "content",
    doc_id_field: str = "_id",
    max_documents: int = MAX_DOCUMENTS,
) -> List[Dict]:
    """
    Extract documents from OpenSearch using PIT/Scroll API.

    Args:
        endpoint: OpenSearch endpoint URL
        index_name: Name of the index to extract from
        text_field: Field name containing the document text
        doc_id_field: Field name for document ID
        max_documents: Maximum number of documents to extract

    Returns:
        List of documents with 'id' and 'text' fields
    """
    logger.info(f"Connecting to OpenSearch: {endpoint}")
    client = get_opensearch_client(endpoint)

    documents = []

    try:
        # Open Point-in-Time for consistent snapshot
        logger.info(f"Opening PIT for index: {index_name}")
        pit_response = client.create_pit(
            index=index_name,
            params={"keep_alive": "5m"}
        )
        pit_id = pit_response["pit_id"]
        logger.info(f"PIT created: {pit_id}")

        # Initial search
        search_body = {
            "size": 1000,
            "query": {"match_all": {}},
            "_source": [text_field],
            "pit": {
                "id": pit_id,
                "keep_alive": "5m"
            },
            "sort": [{"_id": "asc"}]
        }

        search_after = None

        while len(documents) < max_documents:
            if search_after:
                search_body["search_after"] = search_after

            logger.info(f"Fetching documents (current count: {len(documents)})")
            response = client.search(body=search_body)

            hits = response["hits"]["hits"]
            if not hits:
                logger.info("No more documents to fetch")
                break

            for hit in hits:
                doc_id = hit["_id"]

                # Extract text field
                text = hit["_source"].get(text_field, "")

                if text:  # Only include documents with text
                    documents.append({
                        "id": doc_id,
                        "text": text
                    })

                if len(documents) >= max_documents:
                    break

            # Update search_after for next iteration
            search_after = hits[-1]["sort"]

        # Close PIT
        logger.info(f"Closing PIT: {pit_id}")
        client.delete_pit(body={"pit_id": pit_id})

    except Exception as e:
        logger.error(f"Error extracting documents: {str(e)}")
        raise

    # Sample if we got more than max_documents
    if len(documents) > max_documents:
        random.seed(42)
        documents = random.sample(documents, max_documents)

    logger.info(f"Extracted {len(documents)} documents")
    return documents


def save_documents_to_s3(documents: List[Dict], bucket: str, key: str) -> str:
    """Save documents to S3 as JSONL."""
    logger.info(f"Saving {len(documents)} documents to s3://{bucket}/{key}")

    # Convert to JSONL format
    jsonl_content = "\n".join(json.dumps(doc) for doc in documents)

    # Upload to S3
    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=jsonl_content.encode("utf-8"),
        ContentType="application/jsonlines"
    )

    s3_path = f"s3://{bucket}/{key}"
    logger.info(f"Documents saved to {s3_path}")
    return s3_path


def handler(event, context):
    """Lambda handler for data extraction."""
    logger.info(f"Event: {json.dumps(event)}")

    try:
        # Extract parameters
        opensearch_endpoint = event["opensearch_endpoint"]
        index_name = event["index_name"]
        text_field = event.get("text_field", "content")
        doc_id_field = event.get("doc_id_field", "_id")
        max_documents = int(event.get("max_documents", MAX_DOCUMENTS))

        # Extract documents from OpenSearch
        documents = extract_documents_from_opensearch(
            endpoint=opensearch_endpoint,
            index_name=index_name,
            text_field=text_field,
            doc_id_field=doc_id_field,
            max_documents=max_documents,
        )

        # Save to S3
        s3_key = "raw-corpus/documents.jsonl"
        s3_path = save_documents_to_s3(documents, DATA_BUCKET, s3_key)

        # Return result
        return {
            "statusCode": 200,
            "status": "success",
            "s3_path": s3_path,
            "document_count": len(documents),
            "data_bucket": DATA_BUCKET,
        }

    except Exception as e:
        logger.error(f"Error in handler: {str(e)}", exc_info=True)
        return {
            "statusCode": 500,
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__,
        }
