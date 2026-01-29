"""
Step Functions State Machine Definition for OpenSearch Model Fine-Tuning
Implements conditional workflow for data preparation, training, and deployment
"""
from aws_cdk import (
    Duration,
    aws_stepfunctions as sfn,
    aws_stepfunctions_tasks as sfn_tasks,
    aws_lambda as lambda_,
    aws_s3 as s3,
    aws_iam as iam,
    aws_ec2 as ec2,
)
from constructs import Construct


def build_state_machine_definition(
    scope: Construct,
    data_extractor_lambda: lambda_.Function,
    s3_validator_lambda: lambda_.Function,
    bedrock_batch_lambda: lambda_.Function,
    register_model_lambda: lambda_.Function,
    data_bucket: s3.Bucket,
    sagemaker_training_role: iam.Role,
    sagemaker_endpoint_role: iam.Role,
    opensearch_remote_inference_role: iam.Role,
    bedrock_batch_role: iam.Role,
    training_instance_type: str,
    inference_instance_type: str,
) -> sfn.IChainable:
    """
    Build the Step Functions state machine for the fine-tuning workflow.

    Workflow:
    1. Data Source Choice (S3 or OpenSearch)
    2. Bedrock Query Generation
    3. SageMaker Training
    4. SageMaker Endpoint Deployment
    5. OpenSearch Model Registration
    """

    # Define error handling - retry parameters to be used with add_retry()
    retry_errors = ["Lambda.ServiceException", "Lambda.TooManyRequestsException"]
    retry_interval = Duration.seconds(2)
    retry_max_attempts = 3
    retry_backoff_rate = 2.0

    # Success state
    success_state = sfn.Succeed(scope, "Success", comment="Fine-tuning pipeline completed successfully")

    # Failure state
    failure_state = sfn.Fail(
        scope,
        "FailureState",
        cause="Pipeline failed",
        error="PipelineExecutionError",
    )

    # 1. Data Source Choice
    data_source_choice = sfn.Choice(scope, "DataSourceChoice", comment="Choose data source: S3 or OpenSearch")

    # 2a. Extract from OpenSearch (if input_type == "opensearch")
    extract_from_opensearch = sfn_tasks.LambdaInvoke(
        scope,
        "ExtractFromOpenSearch",
        lambda_function=data_extractor_lambda,
        payload=sfn.TaskInput.from_object({
            "opensearch_endpoint": sfn.JsonPath.string_at("$.opensearch_endpoint"),
            "index_name": sfn.JsonPath.string_at("$.index_name"),
            "text_field": sfn.JsonPath.string_at("$.text_field"),
            "doc_id_field": sfn.JsonPath.string_at("$.doc_id_field"),
            "max_documents": sfn.JsonPath.string_at("$.max_documents"),
        }),
        result_path="$.extraction_result",
        output_path="$",
    ).add_retry(
        errors=retry_errors,
        interval=retry_interval,
        max_attempts=retry_max_attempts,
        backoff_rate=retry_backoff_rate,
    )

    # 2b. Validate S3 Corpus (if input_type == "s3")
    validate_s3_corpus = sfn_tasks.LambdaInvoke(
        scope,
        "ValidateS3Corpus",
        lambda_function=s3_validator_lambda,
        payload=sfn.TaskInput.from_object({
            "s3_corpus_path": sfn.JsonPath.string_at("$.s3_corpus_path"),
            "max_documents": sfn.JsonPath.string_at("$.max_documents"),
        }),
        result_path="$.validation_result",
        output_path="$",
    ).add_retry(
        errors=retry_errors,
        interval=retry_interval,
        max_attempts=retry_max_attempts,
        backoff_rate=retry_backoff_rate,
    )

    # 3. Normalize data paths from both sources
    # This ensures both OpenSearch and S3 paths are available in a common location
    normalize_opensearch_path = sfn.Pass(
        scope,
        "NormalizeOpenSearchPath",
        parameters={
            "documents_s3_path": sfn.JsonPath.string_at("$.extraction_result.Payload.s3_path"),
            "input_type.$": "$.input_type",
            "data_bucket.$": "$.data_bucket",
            "extraction_result.$": "$.extraction_result",
        },
        output_path="$",
    )

    normalize_s3_path = sfn.Pass(
        scope,
        "NormalizeS3Path",
        parameters={
            "documents_s3_path": sfn.JsonPath.string_at("$.validation_result.Payload.s3_path"),
            "input_type.$": "$.input_type",
            "data_bucket.$": "$.data_bucket",
            "validation_result.$": "$.validation_result",
        },
        output_path="$",
    )

    # 4. Prepare Bedrock Input
    prepare_bedrock_input = sfn_tasks.LambdaInvoke(
        scope,
        "PrepareBedrockInput",
        lambda_function=bedrock_batch_lambda,
        payload=sfn.TaskInput.from_object({
            "operation": "prepare_input",
            "data_source": sfn.JsonPath.string_at("$.input_type"),
            "s3_documents_path": sfn.JsonPath.string_at("$.documents_s3_path"),
        }),
        result_path="$.bedrock_input_result",
        output_path="$",
    ).add_retry(
        errors=retry_errors,
        interval=retry_interval,
        max_attempts=retry_max_attempts,
        backoff_rate=retry_backoff_rate,
    )

    # 4. Create Bedrock Batch Job
    create_bedrock_batch_job = sfn_tasks.LambdaInvoke(
        scope,
        "CreateBedrockBatchJob",
        lambda_function=bedrock_batch_lambda,
        payload=sfn.TaskInput.from_object({
            "operation": "create_job",
            "s3_input_path": sfn.JsonPath.string_at("$.bedrock_input_result.Payload.s3_input_path"),
            "bedrock_batch_role_arn": bedrock_batch_role.role_arn,
        }),
        result_path="$.bedrock_job_result",
        output_path="$",
    ).add_retry(
        errors=retry_errors,
        interval=retry_interval,
        max_attempts=retry_max_attempts,
        backoff_rate=retry_backoff_rate,
    )

    # 5. Wait for Bedrock Job
    wait_for_bedrock = sfn.Wait(
        scope,
        "WaitForBedrockJob",
        time=sfn.WaitTime.duration(Duration.seconds(60)),
    )

    # 6. Check Bedrock Job Status
    check_bedrock_status = sfn_tasks.LambdaInvoke(
        scope,
        "CheckBedrockJobStatus",
        lambda_function=bedrock_batch_lambda,
        payload=sfn.TaskInput.from_object({
            "operation": "check_status",
            "job_id": sfn.JsonPath.string_at("$.bedrock_job_result.Payload.job_id"),
        }),
        result_path="$.bedrock_status_result",
        output_path="$",
    ).add_retry(
        errors=retry_errors,
        interval=retry_interval,
        max_attempts=retry_max_attempts,
        backoff_rate=retry_backoff_rate,
    )

    # 7. Bedrock Status Choice
    bedrock_status_choice = sfn.Choice(scope, "BedrockStatusChoice", comment="Check if Bedrock job is complete")

    # 8. Process Bedrock Output
    process_bedrock_output = sfn_tasks.LambdaInvoke(
        scope,
        "ProcessBedrockOutput",
        lambda_function=bedrock_batch_lambda,
        payload=sfn.TaskInput.from_object({
            "operation": "process_output",
            "job_id": sfn.JsonPath.string_at("$.bedrock_job_result.Payload.job_id"),
            "output_s3_path": sfn.JsonPath.string_at("$.bedrock_status_result.Payload.output_s3_path"),
        }),
        result_path="$.training_data_result",
        output_path="$",
    ).add_retry(
        errors=retry_errors,
        interval=retry_interval,
        max_attempts=retry_max_attempts,
        backoff_rate=retry_backoff_rate,
    )

    # 9. Start SageMaker Training Job
    start_sagemaker_training = sfn_tasks.SageMakerCreateTrainingJob(
        scope,
        "StartSageMakerTraining",
        training_job_name=sfn.JsonPath.string_at(
            "States.Format('finetune-{}', $.model_name)"
        ),
        algorithm_specification=sfn_tasks.AlgorithmSpecification(
            training_image=sfn_tasks.DockerImage.from_registry(
                "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.1.0-gpu-py310"
            ),
            training_input_mode=sfn_tasks.InputMode.FILE,
        ),
        input_data_config=[
            sfn_tasks.Channel(
                channel_name="training",
                data_source=sfn_tasks.DataSource(
                    s3_data_source=sfn_tasks.S3DataSource(
                        s3_location=sfn_tasks.S3Location.from_json_expression(
                            "$.training_data_result.Payload.training_data_s3_path"
                        ),
                        s3_data_type=sfn_tasks.S3DataType.S3_PREFIX,
                    )
                ),
            )
        ],
        output_data_config=sfn_tasks.OutputDataConfig(
            s3_output_location=sfn_tasks.S3Location.from_bucket(
                data_bucket,
                "model-artifacts",
            )
        ),
        resource_config=sfn_tasks.ResourceConfig(
            instance_count=1,
            instance_type=ec2.InstanceType(training_instance_type),
            volume_size=cdk.Size.gibibytes(50),
        ),
        stopping_condition=sfn_tasks.StoppingCondition(
            max_runtime=Duration.hours(2)
        ),
        role=sagemaker_training_role,
        hyperparameters={
            "model-id": sfn.JsonPath.string_at("$.base_model_id"),
            "model-type": sfn.JsonPath.string_at("$.model_type"),
            "num-epochs": "3",
            "batch-size": "32",
            "learning-rate": "2e-5",
            "max-seq-length": "512",
        },
        result_path="$.training_job_result",
        integration_pattern=sfn.IntegrationPattern.RUN_JOB,
    )

    # 10. Create SageMaker Model
    create_sagemaker_model = sfn_tasks.CallAwsService(
        scope,
        "CreateSageMakerModel",
        service="sagemaker",
        action="createModel",
        parameters={
            "ModelName": sfn.JsonPath.string_at("$.model_name"),
            "PrimaryContainer": {
                "Image": "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.1.0-gpu-py310",
                "ModelDataUrl": sfn.JsonPath.string_at("$.training_job_result.ModelArtifacts.S3ModelArtifacts"),
            },
            "ExecutionRoleArn": sagemaker_endpoint_role.role_arn,
        },
        result_path="$.model_result",
        iam_resources=["*"],
    )

    # 11. Create Endpoint Config
    create_endpoint_config = sfn_tasks.CallAwsService(
        scope,
        "CreateEndpointConfig",
        service="sagemaker",
        action="createEndpointConfig",
        parameters={
            "EndpointConfigName": sfn.JsonPath.string_at(
                "States.Format('{}-config', $.model_name)"
            ),
            "ProductionVariants": [
                {
                    "VariantName": "AllTraffic",
                    "ModelName": sfn.JsonPath.string_at("$.model_name"),
                    "InitialInstanceCount": 1,
                    "InstanceType": inference_instance_type,
                }
            ],
        },
        result_path="$.endpoint_config_result",
        iam_resources=["*"],
    )

    # 12. Create Endpoint
    create_endpoint = sfn_tasks.CallAwsService(
        scope,
        "CreateEndpoint",
        service="sagemaker",
        action="createEndpoint",
        parameters={
            "EndpointName": sfn.JsonPath.string_at(
                "States.Format('{}-endpoint', $.model_name)"
            ),
            "EndpointConfigName": sfn.JsonPath.string_at(
                "States.Format('{}-config', $.model_name)"
            ),
        },
        result_path="$.endpoint_result",
        iam_resources=["*"],
    )

    # 13. Wait for Endpoint
    wait_for_endpoint = sfn.Wait(
        scope,
        "WaitForEndpoint",
        time=sfn.WaitTime.duration(Duration.seconds(60)),
    )

    # 14. Check Endpoint Status
    check_endpoint_status = sfn_tasks.CallAwsService(
        scope,
        "CheckEndpointStatus",
        service="sagemaker",
        action="describeEndpoint",
        parameters={
            "EndpointName": sfn.JsonPath.string_at(
                "States.Format('{}-endpoint', $.model_name)"
            ),
        },
        result_path="$.endpoint_status_result",
        iam_resources=["*"],
    )

    # 15. Endpoint Status Choice
    endpoint_status_choice = sfn.Choice(scope, "EndpointStatusChoice", comment="Check if endpoint is in service")

    # 16. Register to OpenSearch
    register_to_opensearch = sfn_tasks.LambdaInvoke(
        scope,
        "RegisterToOpenSearch",
        lambda_function=register_model_lambda,
        payload=sfn.TaskInput.from_object({
            "opensearch_endpoint": sfn.JsonPath.string_at("$.opensearch_endpoint"),
            "sagemaker_endpoint_name": sfn.JsonPath.string_at(
                "States.Format('{}-endpoint', $.model_name)"
            ),
            "model_name": sfn.JsonPath.string_at("$.model_name"),
            "model_type": sfn.JsonPath.string_at("$.model_type"),
            "embedding_dimension": sfn.JsonPath.string_at("$.embedding_dimension"),
            "region": sfn.JsonPath.string_at("$$.State.Region"),
            "opensearch_remote_inference_role_arn": opensearch_remote_inference_role.role_arn,
        }),
        result_path="$.registration_result",
        output_path="$",
    ).add_retry(
        errors=retry_errors,
        interval=retry_interval,
        max_attempts=retry_max_attempts,
        backoff_rate=retry_backoff_rate,
    )

    # Build the workflow chain
    # Data source branching
    data_source_choice.when(
        sfn.Condition.string_equals("$.input_type", "opensearch"),
        extract_from_opensearch,
    ).when(
        sfn.Condition.string_equals("$.input_type", "s3"),
        validate_s3_corpus,
    ).otherwise(failure_state)

    # Connect both paths through normalization to Bedrock preparation
    extract_from_opensearch.next(normalize_opensearch_path)
    validate_s3_corpus.next(normalize_s3_path)
    normalize_opensearch_path.next(prepare_bedrock_input)
    normalize_s3_path.next(prepare_bedrock_input)

    # Bedrock workflow with polling
    prepare_bedrock_input.next(create_bedrock_batch_job).next(wait_for_bedrock).next(check_bedrock_status).next(
        bedrock_status_choice
    )

    bedrock_status_choice.when(
        sfn.Condition.string_equals("$.bedrock_status_result.Payload.status", "Completed"),
        process_bedrock_output,
    ).when(
        sfn.Condition.string_equals("$.bedrock_status_result.Payload.status", "InProgress"),
        wait_for_bedrock,
    ).when(
        sfn.Condition.string_equals("$.bedrock_status_result.Payload.status", "Failed"),
        failure_state,
    ).otherwise(wait_for_bedrock)

    # Training and deployment workflow
    process_bedrock_output.next(start_sagemaker_training).next(create_sagemaker_model).next(
        create_endpoint_config
    ).next(create_endpoint).next(wait_for_endpoint).next(check_endpoint_status).next(endpoint_status_choice)

    # Endpoint status polling
    endpoint_status_choice.when(
        sfn.Condition.string_equals("$.endpoint_status_result.EndpointStatus", "InService"),
        register_to_opensearch,
    ).when(
        sfn.Condition.string_equals("$.endpoint_status_result.EndpointStatus", "Creating"),
        wait_for_endpoint,
    ).when(
        sfn.Condition.string_equals("$.endpoint_status_result.EndpointStatus", "Failed"),
        failure_state,
    ).otherwise(wait_for_endpoint)

    # Final registration leads to success
    register_to_opensearch.next(success_state)

    # Return the starting state
    return data_source_choice


# Import cdk at module level for Size
import aws_cdk as cdk
