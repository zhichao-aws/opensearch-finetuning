# OpenSearch Fine-Tuning

Automated pipeline for fine-tuning retrieval models using Bedrock synthetic data generation and SageMaker training. Deploys as a single CloudFormation stack with a Step Functions workflow.

## Pipeline Overview

1. **Data Extraction** — Extract documents from OpenSearch index or validate existing S3 corpus
2. **Query Generation** — Generate synthetic queries via Bedrock batch inference
3. **Hard Negative Mining** — Build BM25 hard negative candidates
4. **Teacher Scoring** — Score query-document pairs with a cross-encoder teacher model
5. **Training** — Fine-tune the embedding model with KL divergence distillation
6. **Deployment** *(optional)* — Deploy to SageMaker endpoint and register in OpenSearch

## Prerequisites

- AWS CLI configured with appropriate permissions
- An OpenSearch domain (if using `opensearch` input type)
- If using OpenSearch input: the IAM role `FineTuning-LambdaInvokeOpenSearchRole` must be mapped to OpenSearch backend roles with index read access and `ml_full_access` before deployment
- **(Recommended)** `ml.p4d.24xlarge` quota for SageMaker training: While not strictly required (the pipeline defaults to `ml.g5.2xlarge`), we strongly recommend requesting quota for `ml.p4d.24xlarge` instances to significantly accelerate model training. You can request this at: **AWS Console > Service Quotas > AWS services > Amazon SageMaker > ml.p4d.24xlarge for training job usage**.

## Deploy

### Quick Start (S3 input)

```bash
# Step 1: Download template from GitHub and upload to your S3 bucket
curl -sL https://raw.githubusercontent.com/zhichao-aws/opensearch-finetuning/main/opensearch-finetune-poc.yaml \
  | aws s3 cp - s3://<YOUR_BUCKET>/cfn/opensearch-finetune-poc.yaml --region <REGION>

# Step 2: Create stack
aws cloudformation create-stack \
  --region <REGION> \
  --stack-name <STACK_NAME> \
  --template-url https://<YOUR_BUCKET>.s3.<REGION>.amazonaws.com/cfn/opensearch-finetune-poc.yaml \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameters \
    ParameterKey=ModelName,ParameterValue=<MODEL_NAME> \
    ParameterKey=InputType,ParameterValue=s3 \
    ParameterKey=S3CorpusPath,ParameterValue=s3://<BUCKET>/<PATH>/corpus.jsonl \
    ParameterKey=OpenSearchEndpoint,ParameterValue=https://<DOMAIN>.<REGION>.es.amazonaws.com \
    ParameterKey=ScoringInstanceType,ParameterValue=ml.p4d.24xlarge \
    ParameterKey=TrainingInstanceType,ParameterValue=ml.p4d.24xlarge \
    ParameterKey=TrainBatchSize,ParameterValue=4
```

### OpenSearch Input

```bash
curl -sL https://raw.githubusercontent.com/zhichao-aws/opensearch-finetuning/main/opensearch-finetune-poc.yaml \
  | aws s3 cp - s3://<YOUR_BUCKET>/cfn/opensearch-finetune-poc.yaml --region <REGION>

aws cloudformation create-stack \
  --region <REGION> \
  --stack-name <STACK_NAME> \
  --template-url https://<YOUR_BUCKET>.s3.<REGION>.amazonaws.com/cfn/opensearch-finetune-poc.yaml \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameters \
    ParameterKey=ModelName,ParameterValue=<MODEL_NAME> \
    ParameterKey=InputType,ParameterValue=opensearch \
    ParameterKey=OpenSearchEndpoint,ParameterValue=https://<DOMAIN>.<REGION>.es.amazonaws.com \
    ParameterKey=OpenSearchIndexName,ParameterValue=<INDEX_NAME> \
    ParameterKey=TextFieldNames,ParameterValue=title\\,text \
    ParameterKey=ScoringInstanceType,ParameterValue=ml.p4d.24xlarge \
    ParameterKey=TrainingInstanceType,ParameterValue=ml.p4d.24xlarge \
    ParameterKey=TrainBatchSize,ParameterValue=4
```

### Training Only (no endpoint deployment)

```bash
curl -sL https://raw.githubusercontent.com/zhichao-aws/opensearch-finetuning/main/opensearch-finetune-poc.yaml \
  | aws s3 cp - s3://<YOUR_BUCKET>/cfn/opensearch-finetune-poc.yaml --region <REGION>

aws cloudformation create-stack \
  --region <REGION> \
  --stack-name <STACK_NAME> \
  --template-url https://<YOUR_BUCKET>.s3.<REGION>.amazonaws.com/cfn/opensearch-finetune-poc.yaml \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameters \
    ParameterKey=ModelName,ParameterValue=<MODEL_NAME> \
    ParameterKey=InputType,ParameterValue=s3 \
    ParameterKey=S3CorpusPath,ParameterValue=s3://<BUCKET>/<PATH>/corpus.jsonl \
    ParameterKey=DeployEndpoint,ParameterValue=false \
    ParameterKey=RegisterConnector,ParameterValue=false \
    ParameterKey=ScoringInstanceType,ParameterValue=ml.p4d.24xlarge \
    ParameterKey=TrainingInstanceType,ParameterValue=ml.p4d.24xlarge \
    ParameterKey=TrainBatchSize,ParameterValue=4
```

### S3 Corpus Format

If using `InputType=s3`, provide a JSONL file where each line has a `text` field:

```jsonl
{"text": "Document content here..."}
{"text": "Another document..."}
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| **ModelName** | *(required)* | Unique name for the fine-tuned model (alphanumeric/hyphens, max 40 chars) |
| **InputType** | `s3` | Data source: `s3` or `opensearch` |
| **DeployEndpoint** | `true` | Whether to deploy the fine-tuned model to a SageMaker endpoint |
| **RegisterConnector** | `true` | Whether to register the SageMaker endpoint as an OpenSearch connector |
| **OpenSearchEndpoint** | | OpenSearch domain endpoint (required if `InputType=opensearch` or `RegisterConnector=true`) |
| **OpenSearchIndexName** | | Index to extract documents from (required if `opensearch`) |
| **TextFieldNames** | `content` | Comma-separated field names for document text |
| **S3CorpusPath** | | S3 path to corpus JSONL (required if `s3`) |
| **BaseModelId** | `BAAI/bge-m3` | HuggingFace model ID for base model |
| **BedrockModelId** | `us.anthropic.claude-haiku-4-5-20251001-v1:0` | Bedrock model for query generation |
| **MaxCorpusDocuments** | `10000000` | Max documents for BM25 hard negative pool |
| **MaxQueryDocuments** | `20000` | Max documents to generate queries for |
| **QueriesPerDocument** | `5` | Synthetic queries per document |
| **TrainingInstanceType** | `ml.g5.2xlarge` | SageMaker training instance |
| **ScoringInstanceType** | `ml.g5.2xlarge` | SageMaker scoring instance |
| **InferenceInstanceType** | `ml.g5.xlarge` | SageMaker inference endpoint instance |
| **MaxSteps** | `500` | Max training steps |
| **LearningRate** | `5e-6` | Training learning rate |
| **TrainBatchSize** | `1` | Per-device training batch size |
| **MaxSeqLength** | `512` | Max sequence length for tokenization |

## Development

### Project Structure

```
├── build.sh                          # Build & upload to GitHub release
├── opensearch-finetune-poc.yaml      # CloudFormation template
├── lambdas/
│   ├── bedrock-orchestrator/index.py # Bedrock batch job orchestration
│   ├── data-extractor/index.py       # OpenSearch document extraction
│   ├── register-model/index.py       # OpenSearch model registration
│   └── s3-validator/index.py         # S3 corpus validation
└── training-script/
    ├── process_output.py             # Bedrock output processing & BM25 mining
    ├── score.py                      # Cross-encoder teacher scoring
    ├── train.py                      # Model fine-tuning
    ├── inference.py                  # SageMaker inference handler
    └── requirements.txt
```

### Build & Release

After modifying source code, run the build script to package and upload artifacts to a GitHub release:

```bash
# Requires: gh CLI (brew install gh && gh auth login)

# Build and upload to default tag v1.0.0
./build.sh

# Build and upload to a specific tag
./build.sh v1.1.0
```

This packages each Lambda directory into a zip, the training scripts into a tarball, and uploads all 5 artifacts to the GitHub release.
