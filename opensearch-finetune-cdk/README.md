# OpenSearch Model Fine-Tuning CDK

Automated AWS CDK solution for fine-tuning OpenSearch retrieval models using synthetic data generation and SageMaker.

## Overview

This CDK application deploys a complete pipeline for:
1. **Data Extraction**: Extract documents from OpenSearch or validate S3 corpus
2. **Query Generation**: Use AWS Bedrock to generate synthetic queries
3. **Model Training**: Fine-tune models with SageMaker using sentence-transformers
4. **Deployment**: Deploy model as SageMaker endpoint
5. **Registration**: Automatically register model with OpenSearch ML Commons

## Architecture

```
User Input (S3 or OpenSearch)
  ↓
Step Functions Orchestration
  ↓
┌─────────────────────────────────────────┐
│ 1. Data Preparation                     │
│    - Extract from OpenSearch (PIT/Scroll)│
│    - Or validate S3 corpus               │
└─────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────┐
│ 2. Bedrock Query Generation             │
│    - Generate synthetic queries          │
│    - Create (query, doc) pairs           │
└─────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────┐
│ 3. SageMaker Training                   │
│    - Fine-tune with contrastive loss     │
│    - Save model artifacts to S3          │
└─────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────┐
│ 4. SageMaker Endpoint Deployment        │
│    - Create real-time inference endpoint │
└─────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────┐
│ 5. OpenSearch Registration              │
│    - Create ML Commons connector         │
│    - Register and deploy model           │
└─────────────────────────────────────────┘
  ↓
Model ID Ready for Use
```

## Features

- **Dual Data Sources**: Support for both OpenSearch index extraction and S3 corpus
- **Model Support**: Dense models (BGE, GTE) and sparse models
- **Automated Pipeline**: Step Functions orchestrates the entire workflow
- **Synthetic Data**: Bedrock generates high-quality training queries
- **Production Ready**: Comprehensive error handling and logging

## Prerequisites

- AWS Account with appropriate permissions
- AWS CDK CLI installed (`npm install -g aws-cdk`)
- Python 3.12+
- Existing OpenSearch domain (or create one)
- AWS Bedrock access (Claude or Nova models)

## Installation

1. **Clone and Setup**:
```bash
cd opensearch-finetune-cdk
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2. **Configure AWS Credentials**:
```bash
aws configure
```

3. **Bootstrap CDK** (first time only):
```bash
cdk bootstrap
```

## Configuration

Edit `cdk.json` or set environment variables:

```json
{
  "context": {
    "opensearch_endpoint": "https://your-domain.us-east-1.es.amazonaws.com",
    "model_type": "dense",
    "base_model_id": "BAAI/bge-base-en-v1.5",
    "training_instance_type": "ml.g5.2xlarge",
    "inference_instance_type": "ml.m5.xlarge",
    "max_documents_poc": 20000,
    "bedrock_model_id": "us.anthropic.claude-haiku-4-5-20251001-v1:0"
  }
}
```

Or use environment variables:
```bash
export OPENSEARCH_ENDPOINT="https://..."
export MODEL_TYPE="dense"
export BASE_MODEL_ID="BAAI/bge-base-en-v1.5"
```

## Deployment

### Synthesize CloudFormation Template

```bash
cdk synth > template.yaml
```

This generates a CloudFormation template that you can deploy via AWS Console or CLI.

### Deploy via CDK

```bash
cdk deploy \
  --parameters OpensearchEndpoint=https://your-domain.us-east-1.es.amazonaws.com \
  --parameters ModelType=dense \
  --parameters BaseModelId=BAAI/bge-base-en-v1.5
```

The deployment creates:
- S3 bucket for data and artifacts
- 4 Lambda functions
- Step Functions state machine
- IAM roles with least-privilege permissions
- CloudWatch log groups

## Usage

### Option 1: OpenSearch Index as Data Source

```bash
aws stepfunctions start-execution \
  --state-machine-arn <StateMachineArn-from-output> \
  --name my-finetune-job-$(date +%s) \
  --input '{
    "input_type": "opensearch",
    "opensearch_endpoint": "https://your-domain.us-east-1.es.amazonaws.com",
    "index_name": "my-corpus-index",
    "text_field": "content",
    "doc_id_field": "_id",
    "max_documents": "20000",
    "model_name": "my-finetuned-model-v1",
    "model_type": "dense",
    "base_model_id": "BAAI/bge-base-en-v1.5",
    "embedding_dimension": "768"
  }'
```

### Option 2: S3 Corpus as Data Source

1. **Upload your corpus** to S3 (JSONL format):
```json
{"id": "1", "text": "Document text here..."}
{"id": "2", "text": "Another document..."}
```

2. **Start execution**:
```bash
aws stepfunctions start-execution \
  --state-machine-arn <StateMachineArn-from-output> \
  --name my-finetune-job-$(date +%s) \
  --input '{
    "input_type": "s3",
    "s3_corpus_path": "s3://my-bucket/corpus/documents.jsonl",
    "max_documents": "20000",
    "model_name": "my-finetuned-model-v1",
    "model_type": "dense",
    "base_model_id": "BAAI/bge-base-en-v1.5",
    "embedding_dimension": "768"
  }'
```

### Monitor Execution

```bash
# Check execution status
aws stepfunctions describe-execution --execution-arn <execution-arn>

# View CloudWatch logs
aws logs tail /aws/lambda/DataExtractorLambda --follow
```

### Verify Model Registration

```bash
# Get model info
curl -X GET "https://<opensearch-endpoint>/_plugins/_ml/models/<model-id>" \
  --aws-sigv4 "aws:amz:us-east-1:es"

# Test inference
curl -X POST "https://<opensearch-endpoint>/_plugins/_ml/models/<model-id>/_predict" \
  -H "Content-Type: application/json" \
  -d '{"text_docs": ["what is machine learning?"]}' \
  --aws-sigv4 "aws:amz:us-east-1:es"
```

## Input Parameters

| Parameter | Required | Description |
|-----------|----------|-------------|
| `input_type` | Yes | "opensearch" or "s3" |
| `opensearch_endpoint` | Yes | OpenSearch domain endpoint |
| `index_name` | If opensearch | Index to extract from |
| `text_field` | No | Field containing text (default: "content") |
| `s3_corpus_path` | If s3 | S3 path to corpus file |
| `max_documents` | No | Max docs to process (default: 20000) |
| `model_name` | Yes | Name for the fine-tuned model |
| `model_type` | Yes | "dense" or "sparse" |
| `base_model_id` | Yes | HuggingFace model ID |
| `embedding_dimension` | Yes | Embedding dimension (e.g., 768) |

## Supported Models

### Dense Models
- `BAAI/bge-base-en-v1.5` (768 dim)
- `BAAI/bge-large-en-v1.5` (1024 dim)
- `BAAI/bge-small-en-v1.5` (384 dim)
- `thenlper/gte-base` (768 dim)
- `sentence-transformers/all-MiniLM-L6-v2` (384 dim)

### Sparse Models
- Contact OpenSearch documentation for supported sparse models

## Cost Estimation (POC with 20000 docs)

- **Bedrock**: ~$60 (20000 docs × 2 queries × 500 tokens × $0.003/1K)
- **SageMaker Training**: ~$1-2 (1-2 hours on ml.g5.2xlarge)
- **SageMaker Inference**: ~$100/month (ml.m5.xlarge running continuously)
- **S3, Lambda, Step Functions**: <$5/month

**Total**: ~$120/month (mostly inference endpoint)

## Cleanup

```bash
# Delete the stack
cdk destroy

# Manually delete:
# - S3 bucket contents (if needed)
# - SageMaker endpoints (to stop charges)
```

## Troubleshooting

### Lambda Timeout
- Data extraction can take time for large indexes
- Default timeout: 15 minutes
- Adjust in stack if needed

### Training Failures
- Check CloudWatch logs: `/aws/sagemaker/TrainingJobs`
- Common issues: OOM (reduce batch size), wrong model ID

### OpenSearch Registration Fails
- Ensure `OpenSearchRemoteInferenceRoleArn` is mapped to `ml_full_access` in OpenSearch
- Check OpenSearch access policies

### Bedrock Access Denied
- Request model access in AWS Console → Bedrock → Model access
- Ensure region supports the selected model

## Development

### Testing Individual Steps

During development, you can test each pipeline step independently without running the entire workflow.

#### Option 1: Test Deployed Lambda Functions (`test_steps.py`)

After deploying the CDK stack, use `test_steps.py` to invoke each Lambda directly:

```bash
# Test data extraction from OpenSearch (small batch for testing)
python test_steps.py extract \
  --endpoint https://your-domain.us-east-1.es.amazonaws.com \
  --index my-corpus-index \
  --max-documents 100

# Test S3 corpus validation
python test_steps.py validate \
  --s3-path s3://your-bucket/corpus/documents.jsonl \
  --max-documents 100

# Test Bedrock query generation (full flow)
python test_steps.py generate-queries \
  --s3-docs-path s3://your-bucket/raw-corpus/documents.jsonl

# Or step-by-step with checkpoints:
python test_steps.py generate-queries \
  --s3-docs-path s3://... \
  --prepare-only  # Just prepare Bedrock input

python test_steps.py generate-queries \
  --s3-docs-path s3://... \
  --create-job-only  # Create job but don't wait

python test_steps.py generate-queries \
  --check-job <job-id> \
  --process-output \
  --s3-docs-path s3://...  # Check status and process output

# Test SageMaker training
python test_steps.py train \
  --training-data s3://your-bucket/training-data/training_data.jsonl \
  --base-model-id BAAI/bge-base-en-v1.5 \
  --instance-type ml.g5.2xlarge

# Test endpoint deployment
python test_steps.py deploy \
  --model-artifacts s3://your-bucket/model-artifacts/model.tar.gz \
  --instance-type ml.m5.xlarge

# Test OpenSearch registration
python test_steps.py register \
  --sagemaker-endpoint my-endpoint \
  --opensearch-endpoint https://your-domain.us-east-1.es.amazonaws.com \
  --model-name my-finetuned-model \
  --embedding-dimension 768

# Test endpoint invocation
python test_steps.py invoke \
  --endpoint-name my-model-endpoint \
  --texts "what is machine learning?" "how to train models"
```

#### Option 2: Local Testing (`test_local.py`)

For faster iteration without AWS calls:

```bash
# Generate sample data for local testing
python test_local.py generate-sample-data --output sample_training.jsonl

# Validate local JSONL files
python test_local.py validate-jsonl --file sample_corpus.jsonl --type corpus
python test_local.py validate-jsonl --file sample_training.jsonl --type training

# Test Lambda handlers locally (requires AWS credentials for S3/OpenSearch)
python test_local.py test-extractor \
  --endpoint https://... \
  --index my-index \
  --max-documents 10

python test_local.py test-validator \
  --s3-path s3://bucket/corpus.jsonl
```

### Run Unit Tests
```bash
pip install -r requirements-dev.txt
pytest tests/
```

### Lint and Format
```bash
black .
flake8 .
mypy .
```

## Architecture Decisions

### Why Step Functions?
- State management for long-running jobs
- Built-in error handling and retries
- Visual workflow monitoring

### Why Bedrock Batch?
- Cost-effective for bulk query generation
- Asynchronous processing
- High quality synthetic data

### Why sentence-transformers?
- Industry-standard library
- Easy fine-tuning with contrastive loss
- SageMaker compatible

## Limitations (POC)

- Sample N=20000 documents for speed
- Only positive pairs (no hard negatives)
- Real-time inference only (no serverless)
- Conservative hyperparameters

## Future Enhancements

- Hard negative mining
- Knowledge distillation with cross-encoder
- Serverless inference support
- Cost estimation before training
- Comprehensive benchmarking
- Support for full dataset training

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests
4. Submit a pull request

## License

See LICENSE file

## Support

For issues and questions:
- GitHub Issues
- AWS Support (for AWS service issues)
- OpenSearch Forums (for OpenSearch-specific questions)

## References

- [OpenSearch ML Commons](https://opensearch.org/docs/latest/ml-commons-plugin/)
- [sentence-transformers Documentation](https://www.sbert.net/)
- [AWS CDK Documentation](https://docs.aws.amazon.com/cdk/)
- [AWS Bedrock](https://aws.amazon.com/bedrock/)
- [SageMaker Training](https://docs.aws.amazon.com/sagemaker/latest/dg/train-model.html)
