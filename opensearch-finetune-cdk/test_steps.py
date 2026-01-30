#!/usr/bin/env python3
"""
Test individual pipeline steps independently.

This utility allows you to test each step of the fine-tuning pipeline
without running the entire Step Functions workflow.

Usage:
    # Test data extraction from OpenSearch
    python test_steps.py extract --endpoint https://... --index my-index

    # Test S3 corpus validation
    python test_steps.py validate --s3-path s3://bucket/corpus.jsonl

    # Test Bedrock query generation (prepare -> create job -> poll -> process)
    python test_steps.py generate-queries --s3-docs-path s3://bucket/raw-corpus/docs.jsonl

    # Test SageMaker training
    python test_steps.py train --training-data s3://bucket/training-data/data.jsonl

    # Test endpoint deployment
    python test_steps.py deploy --model-artifacts s3://bucket/model/model.tar.gz

    # Test OpenSearch model registration
    python test_steps.py register --sagemaker-endpoint my-endpoint --opensearch-endpoint https://...
"""

import argparse
import boto3
import json
import sys
import time
from datetime import datetime


def get_lambda_client():
    return boto3.client("lambda")


def get_sfn_client():
    return boto3.client("stepfunctions")


def get_sagemaker_client():
    return boto3.client("sagemaker")


def get_bedrock_client():
    return boto3.client("bedrock")


def invoke_lambda(function_name: str, payload: dict) -> dict:
    """Invoke a Lambda function and return the parsed response."""
    client = get_lambda_client()

    print(f"\n{'='*60}")
    print(f"Invoking Lambda: {function_name}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    print(f"{'='*60}\n")

    response = client.invoke(
        FunctionName=function_name,
        InvocationType="RequestResponse",
        Payload=json.dumps(payload),
    )

    response_payload = json.loads(response["Payload"].read())

    print(f"\nResponse:")
    print(json.dumps(response_payload, indent=2))

    return response_payload


def get_stack_outputs(stack_name: str = "OpenSearchFineTuneStack") -> dict:
    """Get CloudFormation stack outputs to find resource names."""
    cfn = boto3.client("cloudformation")

    try:
        response = cfn.describe_stacks(StackName=stack_name)
        outputs = {}
        for output in response["Stacks"][0].get("Outputs", []):
            outputs[output["OutputKey"]] = output["OutputValue"]
        return outputs
    except Exception as e:
        print(f"Warning: Could not get stack outputs: {e}")
        print("You may need to provide function names manually.")
        return {}


def test_extraction(args):
    """Test data extraction from OpenSearch."""
    print("\n" + "="*60)
    print("TESTING: Data Extraction from OpenSearch")
    print("="*60)

    stack_outputs = get_stack_outputs()
    function_name = args.function_name or stack_outputs.get("DataExtractorLambdaArn", "").split(":")[-1]

    if not function_name:
        print("Error: Could not determine Lambda function name.")
        print("Please provide --function-name or deploy the CDK stack first.")
        return 1

    payload = {
        "opensearch_endpoint": args.endpoint,
        "index_name": args.index,
        "text_field": args.text_field,
        "doc_id_field": args.doc_id_field,
        "max_documents": args.max_documents,
    }

    result = invoke_lambda(function_name, payload)

    if result.get("status") == "success":
        print(f"\n✓ Extraction successful!")
        print(f"  Documents extracted: {result.get('document_count')}")
        print(f"  S3 path: {result.get('s3_path')}")
        return 0
    else:
        print(f"\n✗ Extraction failed: {result.get('error')}")
        return 1


def test_validation(args):
    """Test S3 corpus validation."""
    print("\n" + "="*60)
    print("TESTING: S3 Corpus Validation")
    print("="*60)

    stack_outputs = get_stack_outputs()
    function_name = args.function_name or stack_outputs.get("S3DataValidatorLambdaArn", "").split(":")[-1]

    if not function_name:
        print("Error: Could not determine Lambda function name.")
        print("Please provide --function-name or deploy the CDK stack first.")
        return 1

    payload = {
        "s3_corpus_path": args.s3_path,
        "max_documents": args.max_documents,
    }

    result = invoke_lambda(function_name, payload)

    if result.get("status") == "success":
        print(f"\n✓ Validation successful!")
        print(f"  Documents validated: {result.get('document_count')}")
        print(f"  S3 path: {result.get('s3_path')}")
        return 0
    else:
        print(f"\n✗ Validation failed: {result.get('error')}")
        return 1


def test_generate_queries(args):
    """Test Bedrock query generation (multi-step process)."""
    print("\n" + "="*60)
    print("TESTING: Bedrock Query Generation")
    print("="*60)

    stack_outputs = get_stack_outputs()
    function_name = args.function_name or stack_outputs.get("BedrockBatchOrchestratorLambdaArn", "").split(":")[-1]
    bedrock_role_arn = args.bedrock_role_arn or stack_outputs.get("BedrockBatchRoleArn")

    if not function_name:
        print("Error: Could not determine Lambda function name.")
        print("Please provide --function-name or deploy the CDK stack first.")
        return 1

    # Step 1: Prepare input
    print("\n--- Step 1/4: Prepare Bedrock Input ---")
    prepare_payload = {
        "operation": "prepare_input",
        "s3_documents_path": args.s3_docs_path,
        "data_source": "test",
    }
    prepare_result = invoke_lambda(function_name, prepare_payload)

    if prepare_result.get("status") != "success":
        print(f"\n✗ Prepare input failed: {prepare_result.get('error')}")
        return 1

    s3_input_path = prepare_result.get("s3_input_path")
    print(f"\n✓ Input prepared: {s3_input_path}")
    print(f"  Record count: {prepare_result.get('record_count')}")

    if args.prepare_only:
        print("\n--prepare-only flag set. Stopping here.")
        return 0

    # Step 2: Create batch job
    print("\n--- Step 2/4: Create Bedrock Batch Job ---")
    create_payload = {
        "operation": "create_job",
        "s3_input_path": s3_input_path,
        "bedrock_batch_role_arn": bedrock_role_arn,
    }
    create_result = invoke_lambda(function_name, create_payload)

    if create_result.get("status") != "success":
        print(f"\n✗ Create job failed: {create_result.get('error')}")
        return 1

    job_id = create_result.get("job_id")
    output_s3_uri = create_result.get("output_s3_uri")
    print(f"\n✓ Job created: {job_id}")
    print(f"  Output URI: {output_s3_uri}")

    if args.create_job_only:
        print("\n--create-job-only flag set. Stopping here.")
        print(f"Resume with: python test_steps.py generate-queries --check-job {job_id} --output-s3-uri {output_s3_uri} --s3-docs-path {args.s3_docs_path}")
        return 0

    # Step 3: Poll for completion
    print("\n--- Step 3/4: Wait for Job Completion ---")
    max_wait_minutes = args.max_wait_minutes
    poll_interval = 60
    elapsed = 0

    while elapsed < max_wait_minutes * 60:
        check_payload = {
            "operation": "check_status",
            "job_id": job_id,
        }
        check_result = invoke_lambda(function_name, check_payload)
        status = check_result.get("status")

        print(f"  Status: {status} (elapsed: {elapsed//60}m {elapsed%60}s)")

        if status == "Completed":
            output_s3_path = check_result.get("output_s3_path")
            print(f"\n✓ Job completed!")
            print(f"  Output path: {output_s3_path}")
            break
        elif status in ["Failed", "Stopped"]:
            print(f"\n✗ Job failed with status: {status}")
            return 1

        print(f"  Waiting {poll_interval}s before next check...")
        time.sleep(poll_interval)
        elapsed += poll_interval
    else:
        print(f"\n✗ Job did not complete within {max_wait_minutes} minutes")
        print(f"Resume with: python test_steps.py generate-queries --check-job {job_id} --output-s3-uri {output_s3_uri} --s3-docs-path {args.s3_docs_path}")
        return 1

    # Step 4: Process output
    print("\n--- Step 4/4: Process Bedrock Output ---")
    process_payload = {
        "operation": "process_output",
        "job_id": job_id,
        "output_s3_path": output_s3_path,
        "s3_documents_path": args.s3_docs_path,
    }
    process_result = invoke_lambda(function_name, process_payload)

    if process_result.get("status") == "success":
        print(f"\n✓ Query generation complete!")
        print(f"  Training pairs: {process_result.get('training_pair_count')}")
        print(f"  Training data path: {process_result.get('training_data_s3_path')}")
        return 0
    else:
        print(f"\n✗ Process output failed: {process_result.get('error')}")
        return 1


def test_check_bedrock_job(args):
    """Check status of an existing Bedrock batch job and optionally process output."""
    print("\n" + "="*60)
    print("TESTING: Check Bedrock Job Status")
    print("="*60)

    stack_outputs = get_stack_outputs()
    function_name = args.function_name or stack_outputs.get("BedrockBatchOrchestratorLambdaArn", "").split(":")[-1]

    # Check status
    check_payload = {
        "operation": "check_status",
        "job_id": args.check_job,
    }
    check_result = invoke_lambda(function_name, check_payload)
    status = check_result.get("status")

    print(f"\nJob Status: {status}")

    if status == "Completed":
        output_s3_path = check_result.get("output_s3_path") or args.output_s3_uri

        if args.process_output and args.s3_docs_path:
            print("\n--- Processing Output ---")
            process_payload = {
                "operation": "process_output",
                "job_id": args.check_job,
                "output_s3_path": output_s3_path,
                "s3_documents_path": args.s3_docs_path,
            }
            process_result = invoke_lambda(function_name, process_payload)

            if process_result.get("status") == "success":
                print(f"\n✓ Query generation complete!")
                print(f"  Training pairs: {process_result.get('training_pair_count')}")
                print(f"  Training data path: {process_result.get('training_data_s3_path')}")
                return 0
            else:
                print(f"\n✗ Process output failed: {process_result.get('error')}")
                return 1
        else:
            print(f"  Output path: {output_s3_path}")
            print("\nTo process output, run:")
            print(f"  python test_steps.py generate-queries --check-job {args.check_job} --process-output --s3-docs-path <original-docs-path>")

    return 0 if status in ["Completed", "InProgress", "Submitted"] else 1


def test_train(args):
    """Test SageMaker training job."""
    print("\n" + "="*60)
    print("TESTING: SageMaker Training")
    print("="*60)

    stack_outputs = get_stack_outputs()
    sagemaker_role = args.sagemaker_role or stack_outputs.get("SageMakerTrainingRoleArn")
    data_bucket = args.data_bucket or stack_outputs.get("DataBucketName")

    if not sagemaker_role:
        print("Error: SageMaker role ARN required. Use --sagemaker-role or deploy CDK stack.")
        return 1

    sagemaker = get_sagemaker_client()

    # Generate unique job name
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    job_name = f"test-finetune-{timestamp}"

    # Parse S3 paths
    training_data_path = args.training_data

    # Determine output path
    if training_data_path.startswith("s3://"):
        parts = training_data_path.replace("s3://", "").split("/")
        bucket = parts[0]
        output_path = f"s3://{bucket}/model-artifacts/{job_name}"
    else:
        output_path = f"s3://{data_bucket}/model-artifacts/{job_name}"

    # Training image (PyTorch 2.1.0 GPU)
    region = boto3.session.Session().region_name
    training_image = f"763104351884.dkr.ecr.{region}.amazonaws.com/pytorch-training:2.1.0-gpu-py310-cu121-ubuntu20.04-sagemaker"

    print(f"Job name: {job_name}")
    print(f"Training data: {training_data_path}")
    print(f"Output path: {output_path}")
    print(f"Instance type: {args.instance_type}")

    training_params = {
        "TrainingJobName": job_name,
        "RoleArn": sagemaker_role,
        "AlgorithmSpecification": {
            "TrainingImage": training_image,
            "TrainingInputMode": "File",
        },
        "HyperParameters": {
            "num-epochs": str(args.epochs),
            "batch-size": str(args.batch_size),
            "learning-rate": str(args.learning_rate),
            "max-seq-length": str(args.max_seq_length),
            "model-id": args.base_model_id,
            "sagemaker_program": "finetune_script.py",
            "sagemaker_submit_directory": f"s3://{data_bucket}/training-scripts/sourcedir.tar.gz",
        },
        "InputDataConfig": [
            {
                "ChannelName": "training",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": training_data_path,
                    }
                },
            }
        ],
        "OutputDataConfig": {
            "S3OutputPath": output_path,
        },
        "ResourceConfig": {
            "InstanceType": args.instance_type,
            "InstanceCount": 1,
            "VolumeSizeInGB": 50,
        },
        "StoppingCondition": {
            "MaxRuntimeInSeconds": 3600 * 4,  # 4 hours max
        },
    }

    print("\nStarting training job...")
    print(json.dumps(training_params, indent=2))

    try:
        sagemaker.create_training_job(**training_params)
        print(f"\n✓ Training job started: {job_name}")
    except Exception as e:
        print(f"\n✗ Failed to start training job: {e}")
        return 1

    if not args.no_wait:
        print("\nWaiting for job completion (check CloudWatch logs for progress)...")
        print(f"Log group: /aws/sagemaker/TrainingJobs")

        while True:
            response = sagemaker.describe_training_job(TrainingJobName=job_name)
            status = response["TrainingJobStatus"]
            secondary = response.get("SecondaryStatus", "")

            print(f"  Status: {status} ({secondary})")

            if status == "Completed":
                model_artifacts = response["ModelArtifacts"]["S3ModelArtifacts"]
                print(f"\n✓ Training completed!")
                print(f"  Model artifacts: {model_artifacts}")
                return 0
            elif status in ["Failed", "Stopped"]:
                failure_reason = response.get("FailureReason", "Unknown")
                print(f"\n✗ Training failed: {failure_reason}")
                return 1

            time.sleep(60)
    else:
        print(f"\n--no-wait flag set. Monitor job status with:")
        print(f"  aws sagemaker describe-training-job --training-job-name {job_name}")

    return 0


def test_deploy(args):
    """Test SageMaker endpoint deployment."""
    print("\n" + "="*60)
    print("TESTING: SageMaker Endpoint Deployment")
    print("="*60)

    stack_outputs = get_stack_outputs()
    sagemaker_role = args.sagemaker_role or stack_outputs.get("SageMakerTrainingRoleArn")

    if not sagemaker_role:
        print("Error: SageMaker role ARN required. Use --sagemaker-role or deploy CDK stack.")
        return 1

    sagemaker = get_sagemaker_client()
    region = boto3.session.Session().region_name

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_name = args.model_name or f"test-model-{timestamp}"
    endpoint_config_name = f"{model_name}-config"
    endpoint_name = f"{model_name}-endpoint"

    # Inference image
    inference_image = f"763104351884.dkr.ecr.{region}.amazonaws.com/pytorch-inference:2.1.0-gpu-py310-cu121-ubuntu20.04-sagemaker"

    print(f"Model name: {model_name}")
    print(f"Model artifacts: {args.model_artifacts}")
    print(f"Instance type: {args.instance_type}")

    # Step 1: Create Model
    print("\n--- Step 1/3: Create SageMaker Model ---")
    try:
        sagemaker.create_model(
            ModelName=model_name,
            PrimaryContainer={
                "Image": inference_image,
                "ModelDataUrl": args.model_artifacts,
            },
            ExecutionRoleArn=sagemaker_role,
        )
        print(f"✓ Model created: {model_name}")
    except Exception as e:
        print(f"✗ Failed to create model: {e}")
        return 1

    # Step 2: Create Endpoint Config
    print("\n--- Step 2/3: Create Endpoint Config ---")
    try:
        sagemaker.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[
                {
                    "VariantName": "AllTraffic",
                    "ModelName": model_name,
                    "InstanceType": args.instance_type,
                    "InitialInstanceCount": 1,
                }
            ],
        )
        print(f"✓ Endpoint config created: {endpoint_config_name}")
    except Exception as e:
        print(f"✗ Failed to create endpoint config: {e}")
        return 1

    # Step 3: Create Endpoint
    print("\n--- Step 3/3: Create Endpoint ---")
    try:
        sagemaker.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name,
        )
        print(f"✓ Endpoint creation started: {endpoint_name}")
    except Exception as e:
        print(f"✗ Failed to create endpoint: {e}")
        return 1

    if not args.no_wait:
        print("\nWaiting for endpoint to be InService...")

        while True:
            response = sagemaker.describe_endpoint(EndpointName=endpoint_name)
            status = response["EndpointStatus"]

            print(f"  Status: {status}")

            if status == "InService":
                print(f"\n✓ Endpoint ready: {endpoint_name}")
                return 0
            elif status in ["Failed", "OutOfService"]:
                failure_reason = response.get("FailureReason", "Unknown")
                print(f"\n✗ Endpoint failed: {failure_reason}")
                return 1

            time.sleep(60)
    else:
        print(f"\n--no-wait flag set. Monitor endpoint status with:")
        print(f"  aws sagemaker describe-endpoint --endpoint-name {endpoint_name}")

    return 0


def test_register(args):
    """Test OpenSearch model registration."""
    print("\n" + "="*60)
    print("TESTING: OpenSearch Model Registration")
    print("="*60)

    stack_outputs = get_stack_outputs()
    function_name = args.function_name or stack_outputs.get("RegisterModelLambdaArn", "").split(":")[-1]
    remote_role_arn = args.remote_role_arn or stack_outputs.get("OpenSearchRemoteInferenceRoleArn")

    if not function_name:
        print("Error: Could not determine Lambda function name.")
        print("Please provide --function-name or deploy the CDK stack first.")
        return 1

    region = boto3.session.Session().region_name

    payload = {
        "opensearch_endpoint": args.opensearch_endpoint,
        "sagemaker_endpoint_name": args.sagemaker_endpoint,
        "model_name": args.model_name,
        "model_type": args.model_type,
        "embedding_dimension": args.embedding_dimension,
        "region": region,
        "opensearch_remote_inference_role_arn": remote_role_arn,
    }

    result = invoke_lambda(function_name, payload)

    if result.get("status") == "success":
        print(f"\n✓ Registration successful!")
        print(f"  Connector ID: {result.get('connector_id')}")
        print(f"  Model ID: {result.get('model_id')}")
        print(f"  Model State: {result.get('model_state')}")
        return 0
    else:
        print(f"\n✗ Registration failed: {result.get('error')}")
        return 1


def test_invoke_endpoint(args):
    """Test invoking a SageMaker endpoint directly."""
    print("\n" + "="*60)
    print("TESTING: SageMaker Endpoint Invocation")
    print("="*60)

    runtime = boto3.client("sagemaker-runtime")

    payload = {
        "inputs": args.texts,
    }

    print(f"Endpoint: {args.endpoint_name}")
    print(f"Input texts: {args.texts}")

    try:
        response = runtime.invoke_endpoint(
            EndpointName=args.endpoint_name,
            ContentType="application/json",
            Body=json.dumps(payload),
        )

        result = json.loads(response["Body"].read())
        print(f"\n✓ Invocation successful!")
        print(f"Response: {json.dumps(result, indent=2)}")
        return 0
    except Exception as e:
        print(f"\n✗ Invocation failed: {e}")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Test individual pipeline steps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Extract command
    extract_parser = subparsers.add_parser("extract", help="Test data extraction from OpenSearch")
    extract_parser.add_argument("--endpoint", required=True, help="OpenSearch endpoint URL")
    extract_parser.add_argument("--index", required=True, help="Index name to extract from")
    extract_parser.add_argument("--text-field", default="content", help="Field containing text")
    extract_parser.add_argument("--doc-id-field", default="_id", help="Field containing doc ID")
    extract_parser.add_argument("--max-documents", type=int, default=100, help="Max documents to extract")
    extract_parser.add_argument("--function-name", help="Lambda function name (auto-detected from stack)")

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Test S3 corpus validation")
    validate_parser.add_argument("--s3-path", required=True, help="S3 path to corpus JSONL file")
    validate_parser.add_argument("--max-documents", type=int, default=100, help="Max documents to validate")
    validate_parser.add_argument("--function-name", help="Lambda function name (auto-detected from stack)")

    # Generate queries command
    gen_parser = subparsers.add_parser("generate-queries", help="Test Bedrock query generation")
    gen_parser.add_argument("--s3-docs-path", help="S3 path to documents JSONL")
    gen_parser.add_argument("--bedrock-role-arn", help="Bedrock batch role ARN")
    gen_parser.add_argument("--function-name", help="Lambda function name (auto-detected from stack)")
    gen_parser.add_argument("--prepare-only", action="store_true", help="Only prepare input, don't create job")
    gen_parser.add_argument("--create-job-only", action="store_true", help="Create job but don't wait")
    gen_parser.add_argument("--check-job", help="Check status of existing job ID")
    gen_parser.add_argument("--output-s3-uri", help="Output S3 URI for job (used with --check-job)")
    gen_parser.add_argument("--process-output", action="store_true", help="Process output after checking job")
    gen_parser.add_argument("--max-wait-minutes", type=int, default=60, help="Max minutes to wait for job")

    # Train command
    train_parser = subparsers.add_parser("train", help="Test SageMaker training")
    train_parser.add_argument("--training-data", required=True, help="S3 path to training data JSONL")
    train_parser.add_argument("--base-model-id", default="BAAI/bge-base-en-v1.5", help="Base model ID")
    train_parser.add_argument("--instance-type", default="ml.g5.2xlarge", help="Training instance type")
    train_parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    train_parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    train_parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate")
    train_parser.add_argument("--max-seq-length", type=int, default=512, help="Max sequence length")
    train_parser.add_argument("--sagemaker-role", help="SageMaker execution role ARN")
    train_parser.add_argument("--data-bucket", help="Data bucket name")
    train_parser.add_argument("--no-wait", action="store_true", help="Don't wait for completion")

    # Deploy command
    deploy_parser = subparsers.add_parser("deploy", help="Test SageMaker endpoint deployment")
    deploy_parser.add_argument("--model-artifacts", required=True, help="S3 path to model.tar.gz")
    deploy_parser.add_argument("--model-name", help="Model name (auto-generated if not provided)")
    deploy_parser.add_argument("--instance-type", default="ml.m5.xlarge", help="Inference instance type")
    deploy_parser.add_argument("--sagemaker-role", help="SageMaker execution role ARN")
    deploy_parser.add_argument("--no-wait", action="store_true", help="Don't wait for endpoint to be ready")

    # Register command
    register_parser = subparsers.add_parser("register", help="Test OpenSearch model registration")
    register_parser.add_argument("--sagemaker-endpoint", required=True, help="SageMaker endpoint name")
    register_parser.add_argument("--opensearch-endpoint", required=True, help="OpenSearch endpoint URL")
    register_parser.add_argument("--model-name", required=True, help="Model name for OpenSearch")
    register_parser.add_argument("--model-type", default="dense", choices=["dense", "sparse"], help="Model type")
    register_parser.add_argument("--embedding-dimension", type=int, default=768, help="Embedding dimension")
    register_parser.add_argument("--remote-role-arn", help="OpenSearch remote inference role ARN")
    register_parser.add_argument("--function-name", help="Lambda function name (auto-detected from stack)")

    # Invoke endpoint command
    invoke_parser = subparsers.add_parser("invoke", help="Test invoking a SageMaker endpoint")
    invoke_parser.add_argument("--endpoint-name", required=True, help="SageMaker endpoint name")
    invoke_parser.add_argument("--texts", nargs="+", required=True, help="Texts to encode")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Route to appropriate test function
    if args.command == "extract":
        return test_extraction(args)
    elif args.command == "validate":
        return test_validation(args)
    elif args.command == "generate-queries":
        if args.check_job:
            return test_check_bedrock_job(args)
        elif not args.s3_docs_path:
            print("Error: --s3-docs-path required for generate-queries")
            return 1
        return test_generate_queries(args)
    elif args.command == "train":
        return test_train(args)
    elif args.command == "deploy":
        return test_deploy(args)
    elif args.command == "register":
        return test_register(args)
    elif args.command == "invoke":
        return test_invoke_endpoint(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
