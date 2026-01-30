#!/usr/bin/env python3
"""
AWS CDK Application for OpenSearch Model Fine-Tuning
Deploys infrastructure for automated retrieval model fine-tuning with OpenSearch integration
"""
import os
import aws_cdk as cdk
from stacks.opensearch_finetune_stack import OpenSearchFineTuneStack


app = cdk.App()

# Get configuration from context or environment variables
opensearch_endpoint = app.node.try_get_context("opensearch_endpoint") or os.getenv("OPENSEARCH_ENDPOINT")
model_type = app.node.try_get_context("model_type") or os.getenv("MODEL_TYPE", "dense")
base_model_id = app.node.try_get_context("base_model_id") or os.getenv("BASE_MODEL_ID", "BAAI/bge-base-en-v1.5")
training_instance_type = app.node.try_get_context("training_instance_type") or os.getenv("TRAINING_INSTANCE_TYPE", "ml.g5.2xlarge")
inference_instance_type = app.node.try_get_context("inference_instance_type") or os.getenv("INFERENCE_INSTANCE_TYPE", "ml.m5.xlarge")
max_documents_poc = int(app.node.try_get_context("max_documents_poc") or os.getenv("MAX_DOCUMENTS_POC", "20000"))
bedrock_model_id = app.node.try_get_context("bedrock_model_id") or os.getenv("BEDROCK_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0")

# Create the main stack
OpenSearchFineTuneStack(
    app,
    "OpenSearchFineTuneStack",
    opensearch_endpoint=opensearch_endpoint,
    model_type=model_type,
    base_model_id=base_model_id,
    training_instance_type=training_instance_type,
    inference_instance_type=inference_instance_type,
    max_documents_poc=max_documents_poc,
    bedrock_model_id=bedrock_model_id,
    description="Automated fine-tuning pipeline for OpenSearch retrieval models",
    env=cdk.Environment(
        account=os.getenv("CDK_DEFAULT_ACCOUNT"),
        region=os.getenv("CDK_DEFAULT_REGION", "us-east-1")
    )
)

app.synth()
