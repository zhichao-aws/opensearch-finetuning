# Design Document: OpenSearch Model Fine-Tuning Pipeline

## System Architecture

```
┌─────────────┐
│   User      │
└──────┬──────┘
       │ Input: S3 path or OpenSearch index
       ▼
┌─────────────────────────────────────────┐
│      Step Functions State Machine       │
│  (Orchestrates entire workflow)         │
└─────────────┬───────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│     Phase 1: Data Preparation           │
├─────────────────────────────────────────┤
│ • DataExtractorLambda                   │
│   - PIT/Scroll API for OpenSearch       │
│   - Sample N=20000 docs                  │
│ • S3DataValidatorLambda                 │
│   - Validate JSONL format               │
│ → Output: s3://bucket/raw-corpus/       │
└─────────────┬───────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│   Phase 2: Synthetic Query Generation   │
├─────────────────────────────────────────┤
│ • BedrockBatchOrchestratorLambda        │
│   - Format docs for Bedrock Batch       │
│   - Create batch inference job          │
│   - Poll until completion               │
│   - Parse: {"queries": [...]}           │
│   - Create (query, doc) pairs           │
│ → Output: s3://bucket/training-data/    │
└─────────────┬───────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│      Phase 3: Model Training            │
├─────────────────────────────────────────┤
│ • SageMaker Training Job                │
│   - finetune_script.py                  │
│   - sentence-transformers               │
│   - MultipleNegativesRankingLoss        │
│   - Training params from input          │
│ → Output: s3://bucket/model-artifacts/  │
└─────────────┬───────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│    Phase 4: Endpoint Deployment         │
├─────────────────────────────────────────┤
│ • Create SageMaker Model                │
│ • Create Endpoint Config                │
│ • Create Endpoint (real-time)           │
│ • Poll until InService                  │
└─────────────┬───────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│   Phase 5: OpenSearch Registration      │
├─────────────────────────────────────────┤
│ • RegisterModelLambda                   │
│   - Create ML Connector                 │
│   - Create Model Group                  │
│   - Register Model                      │
│   - Deploy Model                        │
│ → Output: Model ID                      │
└─────────────┬───────────────────────────┘
              │
              ▼
        Model Ready
```

## Component Design

### 1. CDK Stack (`opensearch_finetune_stack.py`)

**Resources Created:**
- S3 Bucket (versioned, encrypted)
- 4 Lambda Functions (Python 3.12, varying memory/timeout)
- Step Functions State Machine
- 6 IAM Roles (least privilege)
- CloudWatch Log Groups

**Key Design Decisions:**
- Single S3 bucket with logical prefixes vs multiple buckets (cost/simplicity)
- Lambda-based processing vs ECS (simplicity for POC)
- Step Functions vs EventBridge (better state management)

### 2. State Machine (`state_machine_definition.py`)

**Flow Control:**
```python
DataSourceChoice
├─ opensearch → ExtractFromOpenSearch
└─ s3 → ValidateS3Corpus
    └─ PrepareBedrockInput
        └─ CreateBedrockBatchJob
            └─ WaitLoop (60s polling)
                └─ StartSageMakerTraining
                    └─ CreateEndpoint
                        └─ WaitLoop (60s polling)
                            └─ RegisterToOpenSearch
```

**Error Handling:**
- Lambda: 3 retries, exponential backoff
- SageMaker: ResourceLimitExceeded retry
- Catch-all failure state with error details

### 3. Lambda Functions

#### DataExtractorLambda
```
Input: {opensearch_endpoint, index_name, text_field, max_documents}
Process:
  1. Connect with IAM auth (AWS4Auth)
  2. Open PIT (5min keep-alive)
  3. Scroll in 1000-doc batches
  4. Sample to max_documents (random.seed=42)
  5. Save as JSONL to S3
Output: {s3_path, document_count}
Memory: 512MB, Timeout: 15min
```

#### BedrockBatchOrchestratorLambda
```
Operations:
  prepare_input: Docs → Bedrock JSONL format
  create_job: Submit batch inference job
  check_status: Poll job status
  process_output: Parse results → training pairs

Prompt: "Generate 2 queries (factoid + conceptual)..."
Output Format: {"query": "...", "document": "..."}
Memory: 256MB, Timeout: 5min
```

#### RegisterModelLambda
```
Process:
  1. Create Connector (SageMaker endpoint → OpenSearch)
  2. Create Model Group
  3. Register Model (with connector_id)
  4. Poll deployment status (30 attempts, 2s interval)
Output: {connector_id, model_id, model_state}
Memory: 512MB, Timeout: 5min
```

### 4. Training Pipeline

#### Training Script (`finetune_script.py`)
```python
# Data loading
InputExample(texts=[query, doc])  # Positive pairs

# Loss function
MultipleNegativesRankingLoss
# Uses in-batch negatives:
# Given batch [(q1,d1), (q2,d2), ...],
# for q1: d1 is positive, d2,d3,... are negatives

# Training
model.fit(
    train_objectives=[(dataloader, loss)],
    epochs=3,
    warmup_steps=10% of total,
    lr=2e-5,
    use_amp=True  # Mixed precision
)
```

**Hyperparameters (POC):**
- Epochs: 3
- Batch Size: 32
- Learning Rate: 2e-5
- Max Seq Length: 512
- Warmup: 10% of steps

#### Inference Handler (`inference/inference.py`)
```python
def model_fn(model_dir):
    return SentenceTransformer(model_dir)

def predict_fn(input_data, model):
    texts = input_data["inputs"]
    embeddings = model.encode(texts, normalize=True)
    return {"embeddings": embeddings.tolist()}
```

## Data Flow

### S3 Bucket Structure
```
s3://bucket/
├── raw-corpus/
│   └── documents.jsonl          # {"id": "...", "text": "..."}
├── bedrock-input/
│   └── batch_input.jsonl        # Bedrock format
├── bedrock-output/
│   └── *.jsonl.out              # Query results
├── training-data/
│   └── training_data.jsonl      # {"query": "...", "document": "..."}
└── model-artifacts/
    └── model.tar.gz             # Trained model
```

## IAM Permissions

### Lambda Execution Role
- S3: GetObject, PutObject, ListBucket
- OpenSearch: ESHttpGet, ESHttpPost, ESHttpPut
- Bedrock: InvokeModel, CreateModelInvocationJob, GetModelInvocationJob
- CloudWatch: Logs

### SageMaker Training Role
- S3: GetObject, PutObject
- ECR: Pull training images
- CloudWatch: Logs

### OpenSearch Remote Inference Role
- SageMaker: InvokeEndpoint
- **User must map this role to `ml_full_access` in OpenSearch**

### Step Functions Role
- Lambda: InvokeFunction
- SageMaker: Create/Describe Training/Model/Endpoint
- IAM: PassRole

## Configuration

### Input Parameters
```json
{
  "input_type": "opensearch|s3",
  "opensearch_endpoint": "https://...",
  "index_name": "...",
  "s3_corpus_path": "s3://...",
  "model_name": "my-model",
  "model_type": "dense|sparse",
  "base_model_id": "BAAI/bge-base-en-v1.5",
  "embedding_dimension": 768,
  "max_documents": 20000
}
```

### Model Support
**Dense:** BGE (base/large/small), GTE, sentence-transformers models
**Sparse:** OpenSearch sparse models (requires custom pooling)

## Performance Characteristics

### POC (20000 docs)
- Data Extraction: 2-5 min
- Bedrock Query Gen: 10-15 min
- Training: 15-30 min (ml.g5.2xlarge)
- Endpoint Deploy: 5-10 min
- **Total: ~35-60 min**

### Cost (POC)
- Bedrock: $15 (10K queries)
- Training: $1-2 (1-2 hours GPU)
- Inference: $100/month (ml.m5.xlarge 24/7)
- Other: $5/month
- **Total: ~$120/month**

## Limitations & Trade-offs

### POC Constraints
- Sample 20000 docs (not full dataset)
- Positive pairs only (no hard negatives)
- Real-time inference (no serverless)
- Conservative hyperparameters
- Single training job (no distributed)

### Design Trade-offs
| Decision | Chosen | Alternative | Rationale |
|----------|--------|-------------|-----------|
| Orchestration | Step Functions | Airflow | Managed, visual, native AWS |
| Query Gen | Bedrock Batch | Online API | Cost-effective for bulk |
| Training | SageMaker | EC2 | Managed, integrated |
| Lambda vs Container | Lambda | ECS/Fargate | Simpler for POC |
| Single vs Multi Bucket | Single | Multiple | Simpler management |

## Future Enhancements (Phase 2)

### Hard Negative Mining
```python
# Before training:
1. Encode corpus with base model
2. For each query, retrieve top-K docs
3. Use non-relevant as hard negatives
4. Train with TripletLoss
```

### Knowledge Distillation
```python
# Use cross-encoder as teacher:
teacher = CrossEncoder('ms-marco-reranker')
scores = teacher.predict([(q, d) for ...])
# Train with KL divergence loss
```

### Serverless Inference
```python
ServerlessConfig(
    max_concurrency=10,
    memory_size_in_mb=4096
)
# Cost: Pay per invocation vs 24/7
```

## Monitoring & Observability

### CloudWatch Metrics
- Step Functions: Execution duration, success/failure rate
- Lambda: Invocations, errors, duration
- SageMaker: Training loss, GPU utilization
- OpenSearch: Model inference latency

### Logs
- Lambda: `/aws/lambda/<function-name>`
- SageMaker: `/aws/sagemaker/TrainingJobs`
- Step Functions: `/aws/stepfunctions/<state-machine>`

## Security

### Encryption
- S3: Server-side encryption (SSE-S3)
- In-transit: TLS 1.2+ for all API calls

### Network
- Lambda: VPC-enabled (optional)
- OpenSearch: IAM auth + role mapping

### Credentials
- No hardcoded credentials
- IAM roles for all services
- OpenSearch: AWS4Auth signature

## Testing Strategy

### Unit Tests
```python
# tests/unit/test_data_extractor.py
@mock_opensearch
def test_extract_documents():
    # Mock OpenSearch client
    # Verify PIT/Scroll logic
    # Check sampling

# tests/unit/test_training_script.py
def test_data_loader():
    # Sample JSONL
    # Verify InputExample creation
```

### Integration Tests
```python
# tests/integration/test_end_to_end.py
def test_full_workflow():
    # Upload test corpus
    # Start Step Functions
    # Poll until complete
    # Verify model in OpenSearch
```

## Deployment

### Prerequisites
- AWS Account
- CDK CLI
- Python 3.12+
- OpenSearch domain

### Commands
```bash
cdk bootstrap              # First time
cdk synth > template.yaml  # Generate CFN
cdk deploy                 # Deploy stack
```

### Rollback
```bash
cdk destroy
# Manually delete SageMaker endpoints
```

## References

- [OpenSearch ML Commons API](https://opensearch.org/docs/latest/ml-commons-plugin/)
- [sentence-transformers Training](https://www.sbert.net/docs/training/overview.html)
- [Bedrock Batch Inference](https://docs.aws.amazon.com/bedrock/latest/userguide/batch-inference.html)
- [SageMaker Training Jobs](https://docs.aws.amazon.com/sagemaker/latest/dg/train-model.html)
