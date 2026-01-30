"""
Main CDK Stack for OpenSearch Model Fine-Tuning Pipeline
Creates S3, IAM, Lambda, Step Functions, and SageMaker resources
"""
import aws_cdk as cdk
from aws_cdk import (
    Stack,
    Duration,
    RemovalPolicy,
    CfnParameter,
    CfnOutput,
    aws_s3 as s3,
    aws_iam as iam,
    aws_lambda as lambda_,
    aws_stepfunctions as sfn,
    aws_logs as logs,
)
from constructs import Construct
from step_functions.state_machine_definition import build_state_machine_definition


class OpenSearchFineTuneStack(Stack):
    """CDK Stack for automated OpenSearch model fine-tuning pipeline"""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        opensearch_endpoint: str = None,
        model_type: str = "dense",
        base_model_id: str = "BAAI/bge-base-en-v1.5",
        training_instance_type: str = "ml.g5.2xlarge",
        inference_instance_type: str = "ml.m5.xlarge",
        max_documents_poc: int = 20000,
        bedrock_model_id: str = "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        **kwargs
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # CloudFormation Parameters
        opensearch_endpoint_param = CfnParameter(
            self,
            "OpensearchEndpoint",
            type="String",
            description="OpenSearch domain endpoint (e.g., https://search-domain.us-east-1.es.amazonaws.com)",
            default=opensearch_endpoint or "",
        )

        model_type_param = CfnParameter(
            self,
            "ModelType",
            type="String",
            description="Model type: dense or sparse",
            allowed_values=["dense", "sparse"],
            default=model_type,
        )

        base_model_id_param = CfnParameter(
            self,
            "BaseModelId",
            type="String",
            description="HuggingFace model ID (e.g., BAAI/bge-base-en-v1.5)",
            default=base_model_id,
        )

        # S3 Bucket for data and artifacts
        data_bucket = s3.Bucket(
            self,
            "DataBucket",
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
            versioned=True,
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
        )

        # IAM Role for Lambda Functions
        lambda_execution_role = iam.Role(
            self,
            "LambdaExecutionRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AWSLambdaBasicExecutionRole"
                )
            ],
        )

        # Grant Lambda access to S3 bucket
        data_bucket.grant_read_write(lambda_execution_role)

        # Grant Lambda access to OpenSearch (ES HTTP actions)
        lambda_execution_role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=[
                    "es:ESHttpGet",
                    "es:ESHttpPost",
                    "es:ESHttpPut",
                    "es:ESHttpDelete",
                ],
                resources=["*"],  # Will be scoped to specific domain by user
            )
        )

        # Grant Lambda access to Bedrock
        lambda_execution_role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=[
                    "bedrock:InvokeModel",
                    "bedrock:CreateModelInvocationJob",
                    "bedrock:GetModelInvocationJob",
                    "bedrock:ListModelInvocationJobs",
                ],
                resources=["*"],
            )
        )

        # Lambda Layer for common dependencies (optional)
        # For production, consider creating a Lambda Layer for opensearch-py

        # Lambda Function: Data Extractor
        data_extractor_lambda = lambda_.Function(
            self,
            "DataExtractorLambda",
            runtime=lambda_.Runtime.PYTHON_3_12,
            handler="index.handler",
            code=lambda_.Code.from_asset("lambda_functions/data_extractor"),
            timeout=Duration.minutes(15),
            memory_size=512,
            environment={
                "DATA_BUCKET": data_bucket.bucket_name,
                "MAX_DOCUMENTS": str(max_documents_poc),
            },
            role=lambda_execution_role,
        )

        # Lambda Function: S3 Data Validator
        s3_validator_lambda = lambda_.Function(
            self,
            "S3DataValidatorLambda",
            runtime=lambda_.Runtime.PYTHON_3_12,
            handler="index.handler",
            code=lambda_.Code.from_asset("lambda_functions/s3_data_validator"),
            timeout=Duration.minutes(5),
            memory_size=256,
            environment={
                "DATA_BUCKET": data_bucket.bucket_name,
            },
            role=lambda_execution_role,
        )

        # Lambda Function: Bedrock Batch Orchestrator
        bedrock_batch_lambda = lambda_.Function(
            self,
            "BedrockBatchOrchestratorLambda",
            runtime=lambda_.Runtime.PYTHON_3_12,
            handler="index.handler",
            code=lambda_.Code.from_asset("lambda_functions/bedrock_batch_orchestrator"),
            timeout=Duration.minutes(5),
            memory_size=256,
            environment={
                "DATA_BUCKET": data_bucket.bucket_name,
                "BEDROCK_MODEL_ID": bedrock_model_id,
            },
            role=lambda_execution_role,
        )

        # IAM Role for Bedrock Batch Jobs
        bedrock_batch_role = iam.Role(
            self,
            "BedrockBatchRole",
            assumed_by=iam.ServicePrincipal("bedrock.amazonaws.com"),
        )
        data_bucket.grant_read_write(bedrock_batch_role)
        bedrock_batch_role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=["bedrock:InvokeModel"],
                resources=["*"],
            )
        )

        # Lambda Function: Register Model to OpenSearch
        register_model_lambda = lambda_.Function(
            self,
            "RegisterModelLambda",
            runtime=lambda_.Runtime.PYTHON_3_12,
            handler="index.handler",
            code=lambda_.Code.from_asset("lambda_functions/register_model"),
            timeout=Duration.minutes(5),
            memory_size=512,
            environment={
                "DATA_BUCKET": data_bucket.bucket_name,
            },
            role=lambda_execution_role,
        )

        # IAM Role for SageMaker Training
        sagemaker_training_role = iam.Role(
            self,
            "SageMakerTrainingRole",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "AmazonSageMakerFullAccess"
                )
            ],
        )
        data_bucket.grant_read_write(sagemaker_training_role)

        # IAM Role for SageMaker Endpoint
        sagemaker_endpoint_role = iam.Role(
            self,
            "SageMakerEndpointRole",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
        )
        sagemaker_endpoint_role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=["sagemaker:InvokeEndpoint"],
                resources=["*"],
            )
        )

        # IAM Role for OpenSearch to invoke SageMaker
        opensearch_remote_inference_role = iam.Role(
            self,
            "OpenSearchRemoteInferenceRole",
            assumed_by=iam.ServicePrincipal("es.amazonaws.com"),
            description="Role for OpenSearch to invoke SageMaker endpoints",
        )
        opensearch_remote_inference_role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=["sagemaker:InvokeEndpoint"],
                resources=["*"],
            )
        )

        # IAM Role for Step Functions
        step_functions_role = iam.Role(
            self,
            "StepFunctionsRole",
            assumed_by=iam.ServicePrincipal("states.amazonaws.com"),
        )

        # Grant Step Functions permission to invoke Lambda functions
        data_extractor_lambda.grant_invoke(step_functions_role)
        s3_validator_lambda.grant_invoke(step_functions_role)
        bedrock_batch_lambda.grant_invoke(step_functions_role)
        register_model_lambda.grant_invoke(step_functions_role)

        # Grant Step Functions permission for SageMaker operations
        step_functions_role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=[
                    "sagemaker:CreateTrainingJob",
                    "sagemaker:DescribeTrainingJob",
                    "sagemaker:StopTrainingJob",
                    "sagemaker:CreateModel",
                    "sagemaker:CreateEndpointConfig",
                    "sagemaker:CreateEndpoint",
                    "sagemaker:DescribeEndpoint",
                    "sagemaker:DeleteEndpoint",
                    "sagemaker:DeleteEndpointConfig",
                    "sagemaker:DeleteModel",
                ],
                resources=["*"],
            )
        )

        # Grant Step Functions permission to pass roles
        step_functions_role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=["iam:PassRole"],
                resources=[
                    sagemaker_training_role.role_arn,
                    sagemaker_endpoint_role.role_arn,
                ],
            )
        )

        # CloudWatch Logs for Step Functions
        state_machine_log_group = logs.LogGroup(
            self,
            "StateMachineLogGroup",
            retention=logs.RetentionDays.ONE_WEEK,
            removal_policy=RemovalPolicy.DESTROY,
        )

        # Build Step Functions State Machine
        state_machine = build_state_machine_definition(
            self,
            data_extractor_lambda=data_extractor_lambda,
            s3_validator_lambda=s3_validator_lambda,
            bedrock_batch_lambda=bedrock_batch_lambda,
            register_model_lambda=register_model_lambda,
            data_bucket=data_bucket,
            sagemaker_training_role=sagemaker_training_role,
            sagemaker_endpoint_role=sagemaker_endpoint_role,
            opensearch_remote_inference_role=opensearch_remote_inference_role,
            bedrock_batch_role=bedrock_batch_role,
            training_instance_type=training_instance_type,
            inference_instance_type=inference_instance_type,
        )

        # Create Step Functions State Machine
        state_machine_resource = sfn.StateMachine(
            self,
            "FineTuneStateMachine",
            definition_body=sfn.DefinitionBody.from_chainable(state_machine),
            role=step_functions_role,
            logs=sfn.LogOptions(
                destination=state_machine_log_group,
                level=sfn.LogLevel.ALL,
            ),
            tracing_enabled=True,
        )

        # Outputs
        CfnOutput(
            self,
            "DataBucketName",
            value=data_bucket.bucket_name,
            description="S3 bucket for data and model artifacts",
        )

        CfnOutput(
            self,
            "StateMachineArn",
            value=state_machine_resource.state_machine_arn,
            description="Step Functions State Machine ARN",
        )

        CfnOutput(
            self,
            "SageMakerTrainingRoleArn",
            value=sagemaker_training_role.role_arn,
            description="SageMaker Training Role ARN",
        )

        CfnOutput(
            self,
            "OpenSearchRemoteInferenceRoleArn",
            value=opensearch_remote_inference_role.role_arn,
            description="OpenSearch Remote Inference Role ARN (map this to ml_full_access in OpenSearch)",
        )

        CfnOutput(
            self,
            "DataExtractorLambdaArn",
            value=data_extractor_lambda.function_arn,
            description="Data Extractor Lambda Function ARN",
        )

        CfnOutput(
            self,
            "RegisterModelLambdaArn",
            value=register_model_lambda.function_arn,
            description="Register Model Lambda Function ARN",
        )
