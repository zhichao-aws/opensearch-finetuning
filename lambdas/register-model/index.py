"""
Register Model Lambda Function

Registers a fine-tuned dense embedding model to OpenSearch ML Commons.
Creates a connector to SageMaker endpoint and registers the model.

Input:
{
    "opensearch_endpoint": "https://search-domain.region.es.amazonaws.com",
    "sagemaker_endpoint_name": "my-model-endpoint",
    "model_name": "my-fine-tuned-model"
}

Output:
{
    "connector_id": "abc123",
    "model_group_id": "def456",
    "model_id": "ghi789"
}
"""

import json
import os
import boto3
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
import urllib.request
import urllib.error
import time


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

    body = json.dumps(data) if data else None
    request = AWSRequest(method=method, url=url, data=body, headers=headers)
    SigV4Auth(credentials, 'es', region).add_auth(request)

    return dict(request.headers), request.data


def make_request(method, url, data=None):
    """Make a signed HTTP request to OpenSearch."""
    headers, signed_body = sign_request(method, url, data)

    req = urllib.request.Request(
        url,
        data=signed_body.encode('utf-8') if signed_body else None,
        headers=headers,
        method=method
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            return json.loads(response.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        error_body = e.read().decode('utf-8')
        print(f"HTTP Error {e.code}: {error_body}")
        raise


def create_connector(opensearch_endpoint, sagemaker_endpoint_name, model_name, region, role_arn):
    """Create a connector to SageMaker endpoint for dense embedding model."""

    connector_body = {
        "name": f"{model_name}-connector",
        "description": f"Connector for fine-tuned dense model: {model_name}",
        "version": 1,
        "protocol": "aws_sigv4",
        "parameters": {
            "region": region,
            "service_name": "sagemaker"
        },
        "credential": {
            "roleArn": role_arn
        },
        "actions": [
            {
                "action_type": "predict",
                "method": "POST",
                "url": f"https://runtime.sagemaker.{region}.amazonaws.com/endpoints/{sagemaker_endpoint_name}/invocations",
                "headers": {
                    "Content-Type": "application/json"
                },
                "request_body": "${parameters.input}",
            }
        ]
    }

    url = f"{opensearch_endpoint}/_plugins/_ml/connectors/_create"
    response = make_request('POST', url, connector_body)

    connector_id = response.get('connector_id')
    print(f"Created connector: {connector_id}")

    return connector_id


def create_model_group(opensearch_endpoint, model_name):
    """Create a model group."""
    model_group_body = {
        "name": f"{model_name}-group",
        "description": f"Model group for fine-tuned model: {model_name}"
    }

    url = f"{opensearch_endpoint}/_plugins/_ml/model_groups/_register"
    response = make_request('POST', url, model_group_body)

    model_group_id = response.get('model_group_id')
    print(f"Created model group: {model_group_id}")

    return model_group_id


def register_model(opensearch_endpoint, model_name, connector_id, model_group_id):
    """Register the dense embedding model with OpenSearch ML Commons."""

    register_body = {
        "name": model_name,
        "function_name": "remote",
        "model_group_id": model_group_id,
        "description": "Fine-tuned dense retrieval model",
        "connector_id": connector_id
    }

    url = f"{opensearch_endpoint}/_plugins/_ml/models/_register"
    response = make_request('POST', url, register_body)

    task_id = response.get('task_id')
    print(f"Register task ID: {task_id}")

    # Wait for registration to complete
    model_id = wait_for_task(opensearch_endpoint, task_id)

    return model_id


def wait_for_task(opensearch_endpoint, task_id, max_attempts=30):
    """Wait for an ML task to complete."""
    url = f"{opensearch_endpoint}/_plugins/_ml/tasks/{task_id}"

    for attempt in range(max_attempts):
        response = make_request('GET', url)
        state = response.get('state')

        print(f"Task {task_id} state: {state}")

        if state == 'COMPLETED':
            return response.get('model_id')
        elif state in ['FAILED', 'CANCELLED']:
            raise Exception(f"Task failed with state: {state}, error: {response.get('error')}")

        time.sleep(2)

    raise Exception(f"Task {task_id} did not complete within timeout")


def deploy_model(opensearch_endpoint, model_id):
    """Deploy the model for inference."""
    url = f"{opensearch_endpoint}/_plugins/_ml/models/{model_id}/_deploy"
    response = make_request('POST', url)

    task_id = response.get('task_id')
    print(f"Deploy task ID: {task_id}")

    # Wait for deployment
    wait_for_task(opensearch_endpoint, task_id)
    print(f"Model {model_id} deployed successfully")


def configure_trusted_connector_endpoints(opensearch_endpoint):
    """Configure ML Commons to trust SageMaker endpoints."""
    settings_body = {
        "persistent": {
            "plugins.ml_commons.trusted_connector_endpoints_regex": [
                "^https://runtime\\.sagemaker\\..*[a-z0-9-]\\.amazonaws\\.com/.*$"
            ]
        }
    }

    url = f"{opensearch_endpoint}/_cluster/settings"
    try:
        response = make_request('PUT', url, settings_body)
        print(f"Configured trusted connector endpoints: {response}")
    except urllib.error.HTTPError as e:
        # If already set or permission issue, log and continue
        print(f"Warning: Could not set trusted connector endpoints: {e}")


def handler(event, context):
    """Lambda handler for model registration."""
    print(f"Event: {json.dumps(event)}")

    # Get parameters
    opensearch_endpoint = event.get('opensearch_endpoint', '').rstrip('/')
    sagemaker_endpoint_name = event.get('sagemaker_endpoint_name')
    model_name = event.get('model_name')

    # Get role ARN from environment or event
    role_arn = event.get('opensearch_remote_inference_role_arn') or os.environ.get('OPENSEARCH_REMOTE_INFERENCE_ROLE_ARN')

    # Get region
    _, region = get_aws_auth()

    # Validate required parameters
    if not opensearch_endpoint:
        raise ValueError("opensearch_endpoint is required")
    if not sagemaker_endpoint_name:
        raise ValueError("sagemaker_endpoint_name is required")
    if not model_name:
        raise ValueError("model_name is required")
    if not role_arn:
        raise ValueError("opensearch_remote_inference_role_arn is required")

    print(f"Registering model {model_name} to {opensearch_endpoint}")
    print(f"SageMaker endpoint: {sagemaker_endpoint_name}")

    # Configure trusted connector endpoints (required for SageMaker)
    configure_trusted_connector_endpoints(opensearch_endpoint)

    # Create connector
    connector_id = create_connector(
        opensearch_endpoint, sagemaker_endpoint_name, model_name, region, role_arn
    )

    # Create model group
    model_group_id = create_model_group(opensearch_endpoint, model_name)

    # Register model
    model_id = register_model(
        opensearch_endpoint, model_name, connector_id, model_group_id
    )

    # Deploy model
    deploy_model(opensearch_endpoint, model_id)

    print(f"Model registered and deployed successfully")
    print(f"Connector ID: {connector_id}")
    print(f"Model Group ID: {model_group_id}")
    print(f"Model ID: {model_id}")

    return {
        "statusCode": 200,
        "connector_id": connector_id,
        "model_group_id": model_group_id,
        "model_id": model_id
    }
