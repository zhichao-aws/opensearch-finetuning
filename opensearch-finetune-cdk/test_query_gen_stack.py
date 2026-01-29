#!/usr/bin/env python3
"""
Minimal CDK Stack for testing query generation workflow
Only deploys S3, Lambda, and IAM resources needed for testing
"""
import aws_cdk as cdk
from aws_cdk import (
    Stack,
    Duration,
    RemovalPolicy,
    CfnOutput,
    aws_s3 as s3,
    aws_iam as iam,
    aws_lambda as lambda_,
)
from constructs import Construct


class TestQueryGenStack(Stack):
    """Minimal stack for testing S3 to Bedrock query generation"""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        bedrock_model_id: str = "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        **kwargs
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # Data bucket for testing
        data_bucket = s3.Bucket(
            self,
            "TestDataBucket",
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
        )

        # IAM Role for Lambda Functions
        lambda_role = iam.Role(
            self,
            "TestLambdaRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AWSLambdaBasicExecutionRole"
                )
            ],
        )

        # Grant Lambda access to S3
        data_bucket.grant_read_write(lambda_role)

        # Grant Lambda access to Bedrock
        lambda_role.add_to_policy(
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

        # IAM Role for Bedrock Batch
        bedrock_batch_role = iam.Role(
            self,
            "TestBedrockBatchRole",
            assumed_by=iam.ServicePrincipal("bedrock.amazonaws.com"),
        )

        # Grant Bedrock access to S3 bucket
        data_bucket.grant_read_write(bedrock_batch_role)

        # S3 Data Validator Lambda
        s3_validator_lambda = lambda_.Function(
            self,
            "TestS3ValidatorLambda",
            runtime=lambda_.Runtime.PYTHON_3_10,
            handler="index.handler",
            code=lambda_.Code.from_asset("lambda_functions/s3_data_validator"),
            timeout=Duration.minutes(5),
            memory_size=512,
            role=lambda_role,
            environment={
                "DATA_BUCKET": data_bucket.bucket_name,
                "MAX_DOCUMENTS": "5000",
            },
        )

        # Bedrock Batch Orchestrator Lambda
        bedrock_lambda = lambda_.Function(
            self,
            "TestBedrockOrchestratorLambda",
            runtime=lambda_.Runtime.PYTHON_3_10,
            handler="index.handler",
            code=lambda_.Code.from_asset("lambda_functions/bedrock_batch_orchestrator"),
            timeout=Duration.minutes(5),
            memory_size=512,
            role=lambda_role,
            environment={
                "DATA_BUCKET": data_bucket.bucket_name,
                "BEDROCK_MODEL_ID": bedrock_model_id,
            },
        )

        # Outputs
        CfnOutput(
            self,
            "DataBucketName",
            value=data_bucket.bucket_name,
            description="S3 bucket for test data",
        )

        CfnOutput(
            self,
            "S3ValidatorLambdaArn",
            value=s3_validator_lambda.function_arn,
            description="S3 Validator Lambda ARN",
        )

        CfnOutput(
            self,
            "BedrockOrchestratorLambdaArn",
            value=bedrock_lambda.function_arn,
            description="Bedrock Orchestrator Lambda ARN",
        )

        CfnOutput(
            self,
            "BedrockBatchRoleArn",
            value=bedrock_batch_role.role_arn,
            description="Bedrock Batch IAM Role ARN",
        )


if __name__ == "__main__":
    app = cdk.App()
    TestQueryGenStack(
        app,
        "TestQueryGenStack",
        env=cdk.Environment(
            account=app.node.try_get_context("account"),
            region=app.node.try_get_context("region") or "us-east-1",
        ),
    )
    app.synth()
