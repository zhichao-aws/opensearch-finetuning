#!/usr/bin/env python3
"""
Test script for S3 to Bedrock query generation workflow
Tests the data validation and query generation steps locally
"""
import json
import sys
import os
import boto3
from datetime import datetime

# Add lambda functions to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lambda_functions/s3_data_validator'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lambda_functions/bedrock_batch_orchestrator'))

from lambda_functions.s3_data_validator.index import validate_and_process_corpus
from lambda_functions.bedrock_batch_orchestrator.index import prepare_bedrock_input, create_bedrock_batch_job


def test_query_generation(
    source_s3_path: str,
    data_bucket: str,
    bedrock_batch_role_arn: str = None,
    max_documents: int = 10,
    dry_run: bool = False
):
    """
    Test the query generation workflow from S3 corpus to Bedrock batch job.

    Args:
        source_s3_path: S3 path to your corpus file (e.g., s3://my-bucket/corpus.txt)
        data_bucket: S3 bucket for intermediate data storage
        bedrock_batch_role_arn: IAM role ARN for Bedrock (required if not dry_run)
        max_documents: Maximum number of documents to process
        dry_run: If True, only prepare input without creating Bedrock job
    """
    print(f"\n{'='*60}")
    print(f"Testing Query Generation Workflow")
    print(f"{'='*60}\n")

    # Set environment variables for Lambda functions
    os.environ['DATA_BUCKET'] = data_bucket
    os.environ['BEDROCK_MODEL_ID'] = 'us.anthropic.claude-haiku-4-5-20251001-v1:0'

    try:
        # Step 1: Validate S3 corpus format
        print(f"Step 1: Validating corpus format from {source_s3_path}")
        print(f"  - Checking JSONL format with required 'id' and 'text' fields")
        print(f"  - Max documents to validate: {max_documents}")

        result1 = validate_and_process_corpus(source_s3_path, max_documents)

        print(f"  ✓ Validated {result1['document_count']} documents")
        print(f"  ✓ Source path: {result1['s3_path']}\n")

        # Step 2: Prepare Bedrock input
        print(f"Step 2: Preparing Bedrock batch input")

        s3_documents_path = result1['s3_path']
        result2 = prepare_bedrock_input(s3_documents_path)

        print(f"  ✓ Created {result2['record_count']} Bedrock input records")
        print(f"  ✓ Saved to: {result2['s3_input_path']}\n")

        # Step 3: Create Bedrock batch job (optional)
        if not dry_run:
            if not bedrock_batch_role_arn:
                print("⚠️  Skipping Bedrock job creation (no role ARN provided)")
                print("   Set bedrock_batch_role_arn parameter to create the job\n")
            else:
                print(f"Step 3: Creating Bedrock batch job")

                result3 = create_bedrock_batch_job(
                    result2['s3_input_path'],
                    bedrock_batch_role_arn
                )

                print(f"  ✓ Created Bedrock job: {result3['job_id']}")
                print(f"  ✓ Job ARN: {result3['job_arn']}")
                print(f"  ✓ Output will be saved to: {result3['output_s3_uri']}\n")

                print("To check job status, run:")
                print(f"  aws bedrock get-model-invocation-job --job-identifier {result3['job_id']}\n")
        else:
            print("Step 3: Skipped (dry run mode)\n")

        print(f"{'='*60}")
        print(f"✅ Test completed successfully!")
        print(f"{'='*60}\n")

        # Print summary
        print("Summary:")
        print(f"  - Documents processed: {result1['document_count']}")
        print(f"  - Bedrock inputs created: {result2['record_count']}")
        print(f"  - Documents location: {result1['s3_path']}")
        print(f"  - Bedrock input location: {result2['s3_input_path']}")

        return {
            'success': True,
            'documents_path': result1['s3_path'],
            'bedrock_input_path': result2['s3_input_path']
        }

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Test query generation workflow from S3 corpus',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run (prepare input only, don't create Bedrock job)
  python test_query_generation.py \\
    --source s3://my-bucket/corpus.txt \\
    --data-bucket my-data-bucket \\
    --max-documents 10 \\
    --dry-run

  # Full run (create Bedrock batch job)
  python test_query_generation.py \\
    --source s3://my-bucket/corpus.txt \\
    --data-bucket my-data-bucket \\
    --role-arn arn:aws:iam::ACCOUNT:role/BedrockBatchRole \\
    --max-documents 10
        """
    )

    parser.add_argument(
        '--source',
        required=True,
        help='S3 path to source corpus file (e.g., s3://bucket/corpus.txt)'
    )
    parser.add_argument(
        '--data-bucket',
        required=True,
        help='S3 bucket for storing intermediate data'
    )
    parser.add_argument(
        '--role-arn',
        help='IAM role ARN for Bedrock batch job (required for full run)'
    )
    parser.add_argument(
        '--max-documents',
        type=int,
        default=10,
        help='Maximum number of documents to process (default: 10)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Prepare input only, do not create Bedrock job'
    )

    args = parser.parse_args()

    test_query_generation(
        source_s3_path=args.source,
        data_bucket=args.data_bucket,
        bedrock_batch_role_arn=args.role_arn,
        max_documents=args.max_documents,
        dry_run=args.dry_run
    )
