"""
Data Extractor Lambda Function

Extracts documents from OpenSearch index using Point-in-Time (PIT) API for consistent,
large-scale document extraction. Supports sampling to limit document count.

Input:
{
    "opensearch_endpoint": "https://search-domain.region.es.amazonaws.com",
    "index_name": "my-index",
    "text_fields": "title,content,description",  # Comma-separated fields to concatenate
    "max_documents": 20000    # Optional, uses env var if not provided
}

Output:
{
    "s3_path": "s3://bucket/raw-corpus/documents.jsonl",
    "document_count": 20000
}
"""

import json
import os
import boto3
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from urllib.parse import urlparse
import urllib.request
import urllib.error


def get_aws_auth():
    """Get AWS credentials for signing requests."""
    session = boto3.Session()
    credentials = session.get_credentials()
    return credentials, session.region_name


def sign_request(method, url, data=None, headers=None):
    """Sign an HTTP request with AWS SigV4."""
    credentials, region = get_aws_auth()

    if headers is None:
        headers = {}
    headers['Content-Type'] = 'application/json'

    request = AWSRequest(method=method, url=url, data=data, headers=headers)
    SigV4Auth(credentials, 'es', region).add_auth(request)

    return dict(request.headers), request.data


def make_request(method, url, data=None):
    """Make a signed HTTP request to OpenSearch."""
    body = json.dumps(data) if data else None
    headers, signed_body = sign_request(method, url, body)

    req = urllib.request.Request(
        url,
        data=signed_body.encode('utf-8') if signed_body else None,
        headers=headers,
        method=method
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.loads(response.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        error_body = e.read().decode('utf-8')
        print(f"HTTP Error {e.code}: {error_body}")
        raise


def create_pit(endpoint, index_name, keep_alive="5m"):
    """Create a Point-in-Time for consistent reads."""
    url = f"{endpoint}/{index_name}/_search/point_in_time?keep_alive={keep_alive}"
    response = make_request('POST', url)
    return response['pit_id']


def delete_pit(endpoint, pit_id):
    """Delete the Point-in-Time."""
    url = f"{endpoint}/_search/point_in_time"
    try:
        make_request('DELETE', url, {"pit_id": pit_id})
    except Exception as e:
        print(f"Warning: Failed to delete PIT: {e}")


def search_with_pit(endpoint, pit_id, text_fields, batch_size=1000, search_after=None):
    """Search using PIT with search_after pagination.

    Args:
        text_fields: List of field names to retrieve from _source
    """
    url = f"{endpoint}/_search"

    query = {
        "size": batch_size,
        "query": {
            "match_all": {}
        },
        "_source": text_fields,
        "pit": {
            "id": pit_id,
            "keep_alive": "5m"
        },
        "sort": [
            {"_doc": "asc"}
        ]
    }

    if search_after:
        query["search_after"] = search_after

    response = make_request('POST', url, query)
    return response


def extract_documents(endpoint, index_name, text_fields, max_documents):
    """Extract documents from OpenSearch using PIT API.

    Args:
        text_fields: List of field names. Values will be concatenated in order with space separator.
    """
    documents = []
    pit_id = None

    try:
        # Create PIT for consistent reads
        print(f"Creating PIT for index: {index_name}")
        pit_id = create_pit(endpoint, index_name)
        print(f"PIT created: {pit_id[:50]}...")

        search_after = None
        batch_num = 0

        while len(documents) < max_documents:
            batch_size = min(1000, max_documents - len(documents))

            response = search_with_pit(
                endpoint, pit_id, text_fields,
                batch_size=batch_size, search_after=search_after
            )

            hits = response.get('hits', {}).get('hits', [])

            if not hits:
                print("No more documents to fetch")
                break

            for hit in hits:
                doc_id = hit['_id']
                source = hit.get('_source', {})

                # Concatenate field values in order, skip empty/missing fields
                text_parts = []
                for field in text_fields:
                    value = source.get(field, '')
                    if value:
                        text_parts.append(str(value))

                text = ' '.join(text_parts)

                if text:  # Only include documents with text
                    documents.append({
                        "id": doc_id,
                        "text": text
                    })

            # Get the sort values from the last hit for pagination
            search_after = hits[-1].get('sort')

            batch_num += 1
            print(f"Batch {batch_num}: Fetched {len(hits)} documents, total: {len(documents)}")

            if len(hits) < batch_size:
                print("Reached end of index")
                break

    finally:
        if pit_id:
            print("Deleting PIT...")
            delete_pit(endpoint, pit_id)

    return documents


def upload_to_s3(documents, bucket, key):
    """Upload documents as JSONL to S3."""
    s3 = boto3.client('s3')

    # Convert to JSONL format
    jsonl_content = '\n'.join(json.dumps(doc, ensure_ascii=False) for doc in documents)

    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=jsonl_content.encode('utf-8'),
        ContentType='application/jsonl'
    )

    return f"s3://{bucket}/{key}"


def handler(event, context):
    """Lambda handler for data extraction."""
    print(f"Event: {json.dumps(event)}")

    # Get parameters
    opensearch_endpoint = event.get('opensearch_endpoint', '').rstrip('/')
    index_name = event.get('index_name')
    max_documents = int(event.get('max_documents', os.environ.get('MAX_DOCUMENTS', 20000)))

    # Parse text_fields - accepts comma-separated string
    text_fields_raw = event.get('text_fields', 'content')
    text_fields = [f.strip() for f in text_fields_raw.split(',') if f.strip()]

    if not text_fields:
        raise ValueError("text_fields must contain at least one field name")

    # Validate required parameters
    if not opensearch_endpoint:
        raise ValueError("opensearch_endpoint is required")
    if not index_name:
        raise ValueError("index_name is required")

    # Get S3 bucket from environment
    bucket = os.environ.get('DATA_BUCKET')
    if not bucket:
        raise ValueError("DATA_BUCKET environment variable is required")

    print(f"Extracting from {opensearch_endpoint}/{index_name}")
    print(f"Text fields: {text_fields}, Max documents: {max_documents}")

    # Extract documents
    documents = extract_documents(
        opensearch_endpoint, index_name, text_fields, max_documents
    )

    print(f"Extracted {len(documents)} documents")

    if not documents:
        raise ValueError(f"No documents found in index {index_name} with fields {text_fields}")

    # Upload to S3
    s3_key = "raw-corpus/documents.jsonl"
    s3_path = upload_to_s3(documents, bucket, s3_key)

    print(f"Uploaded to {s3_path}")

    return {
        "statusCode": 200,
        "s3_path": s3_path,
        "document_count": len(documents)
    }
