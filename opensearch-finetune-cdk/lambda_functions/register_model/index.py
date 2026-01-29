"""
Lambda Function: Register Model to OpenSearch
Registers trained SageMaker model with OpenSearch ML Commons
"""
import json
import logging
import os
from typing import Dict

import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth

logger = logging.getLogger()
logger.setLevel(logging.INFO)


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


def create_connector(
    client: OpenSearch,
    model_name: str,
    endpoint_name: str,
    region: str,
    role_arn: str
) -> str:
    """
    Create ML Commons connector for SageMaker endpoint.

    Args:
        client: OpenSearch client
        model_name: Name for the model
        endpoint_name: SageMaker endpoint name
        region: AWS region
        role_arn: IAM role ARN for OpenSearch to invoke SageMaker

    Returns:
        Connector ID
    """
    logger.info(f"Creating connector for endpoint: {endpoint_name}")

    connector_body = {
        "name": f"{model_name}-connector",
        "description": f"Connector for fine-tuned model {model_name}",
        "version": "1",
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
                "url": f"https://runtime.sagemaker.{region}.amazonaws.com/endpoints/{endpoint_name}/invocations",
                "headers": {
                    "content-type": "application/json"
                },
                "request_body": '{"inputs": "${parameters.inputs}"}'
            }
        ]
    }

    try:
        response = client.transport.perform_request(
            method="POST",
            url="/_plugins/_ml/connectors/_create",
            body=connector_body
        )

        connector_id = response["connector_id"]
        logger.info(f"Created connector: {connector_id}")
        return connector_id

    except Exception as e:
        logger.error(f"Error creating connector: {str(e)}", exc_info=True)
        raise


def create_model_group(client: OpenSearch, model_name: str) -> str:
    """
    Create ML Commons model group.

    Args:
        client: OpenSearch client
        model_name: Name for the model

    Returns:
        Model group ID
    """
    logger.info(f"Creating model group for: {model_name}")

    model_group_body = {
        "name": f"{model_name}-group",
        "description": f"Model group for {model_name}"
    }

    try:
        response = client.transport.perform_request(
            method="POST",
            url="/_plugins/_ml/model_groups/_register",
            body=model_group_body
        )

        model_group_id = response["model_group_id"]
        logger.info(f"Created model group: {model_group_id}")
        return model_group_id

    except Exception as e:
        logger.error(f"Error creating model group: {str(e)}", exc_info=True)
        raise


def register_model(
    client: OpenSearch,
    model_name: str,
    connector_id: str,
    model_group_id: str,
    model_type: str,
    embedding_dimension: int
) -> str:
    """
    Register model with ML Commons.

    Args:
        client: OpenSearch client
        model_name: Name for the model
        connector_id: Connector ID
        model_group_id: Model group ID
        model_type: Model type (dense or sparse)
        embedding_dimension: Embedding dimension

    Returns:
        Model ID
    """
    logger.info(f"Registering model: {model_name}")

    model_body = {
        "name": model_name,
        "function_name": "remote",
        "description": f"Fine-tuned {model_type} retrieval model",
        "model_group_id": model_group_id,
        "connector_id": connector_id
    }

    try:
        # Register model
        response = client.transport.perform_request(
            method="POST",
            url="/_plugins/_ml/models/_register",
            body=model_body,
            params={"deploy": "true"}
        )

        model_id = response["model_id"]
        task_id = response.get("task_id")
        logger.info(f"Registered model: {model_id}, task: {task_id}")

        # Wait for deployment (poll task status)
        import time
        max_attempts = 30
        for attempt in range(max_attempts):
            if task_id:
                task_response = client.transport.perform_request(
                    method="GET",
                    url=f"/_plugins/_ml/tasks/{task_id}"
                )
                state = task_response.get("state")
                logger.info(f"Model deployment state: {state}")

                if state == "COMPLETED":
                    logger.info("Model deployed successfully")
                    break
                elif state == "FAILED":
                    error = task_response.get("error", "Unknown error")
                    raise RuntimeError(f"Model deployment failed: {error}")

            time.sleep(2)

        return model_id

    except Exception as e:
        logger.error(f"Error registering model: {str(e)}", exc_info=True)
        raise


def handler(event, context):
    """Lambda handler for model registration."""
    logger.info(f"Event: {json.dumps(event)}")

    try:
        # Extract parameters
        opensearch_endpoint = event["opensearch_endpoint"]
        sagemaker_endpoint_name = event["sagemaker_endpoint_name"]
        model_name = event["model_name"]
        model_type = event["model_type"]
        embedding_dimension = int(event.get("embedding_dimension", 768))
        region = event["region"]
        opensearch_remote_inference_role_arn = event["opensearch_remote_inference_role_arn"]

        # Create OpenSearch client
        client = get_opensearch_client(opensearch_endpoint)

        # Create connector
        connector_id = create_connector(
            client=client,
            model_name=model_name,
            endpoint_name=sagemaker_endpoint_name,
            region=region,
            role_arn=opensearch_remote_inference_role_arn
        )

        # Create model group
        model_group_id = create_model_group(client, model_name)

        # Register model
        model_id = register_model(
            client=client,
            model_name=model_name,
            connector_id=connector_id,
            model_group_id=model_group_id,
            model_type=model_type,
            embedding_dimension=embedding_dimension
        )

        logger.info(f"Successfully registered model {model_id}")

        return {
            "statusCode": 200,
            "status": "success",
            "connector_id": connector_id,
            "model_group_id": model_group_id,
            "model_id": model_id,
            "model_state": "deployed"
        }

    except Exception as e:
        logger.error(f"Error in handler: {str(e)}", exc_info=True)
        return {
            "statusCode": 500,
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__,
        }
