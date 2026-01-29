# Quick Start Guide

## 5-Minute Setup

### 1. Install Dependencies

```bash
cd opensearch-finetune-cdk
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Bootstrap CDK (First Time Only)

```bash
# Set your AWS credentials
export AWS_DEFAULT_REGION=us-east-1
aws configure

# Bootstrap CDK
cdk bootstrap
```

### 3. Synthesize CloudFormation Template

```bash
cdk synth > template.yaml
```

This creates a CloudFormation template that you can:
- Deploy via AWS Console (CloudFormation → Create Stack → Upload template)
- Deploy via AWS CLI
- Deploy via CDK

### 4. Deploy Stack

**Option A: Via CDK CLI**
```bash
cdk deploy \
  --parameters OpensearchEndpoint=https://your-domain.us-east-1.es.amazonaws.com \
  --parameters ModelType=dense \
  --parameters BaseModelId=BAAI/bge-base-en-v1.5
```

**Option B: Via AWS Console**
1. Go to CloudFormation console
2. Create Stack → Upload template file
3. Upload `template.yaml`
4. Fill in parameters:
   - OpensearchEndpoint: Your OpenSearch domain URL
   - ModelType: `dense`
   - BaseModelId: `BAAI/bge-base-en-v1.5`
5. Create stack

### 5. Get Stack Outputs

After deployment completes:

```bash
# Get outputs
aws cloudformation describe-stacks \
  --stack-name OpenSearchFineTuneStack \
  --query 'Stacks[0].Outputs'
```

You'll see:
- `StateMachineArn`: Step Functions state machine ARN
- `DataBucketName`: S3 bucket for data
- `OpenSearchRemoteInferenceRoleArn`: IAM role ARN (important!)

### 6. Configure OpenSearch (CRITICAL)

Map the IAM role to OpenSearch's ml_full_access:

```bash
# Get the role ARN from stack outputs
ROLE_ARN=$(aws cloudformation describe-stacks \
  --stack-name OpenSearchFineTuneStack \
  --query 'Stacks[0].Outputs[?OutputKey==`OpenSearchRemoteInferenceRoleArn`].OutputValue' \
  --output text)

# Add role mapping to OpenSearch
curl -X PUT "https://your-opensearch-domain/_plugins/_security/api/rolesmapping/ml_full_access" \
  -H 'Content-Type: application/json' \
  -u 'admin:password' \
  -d "{
    \"backend_roles\": [\"$ROLE_ARN\"]
  }"
```

### 7. Start Fine-Tuning Job

**Example with OpenSearch Index:**

```bash
STATE_MACHINE_ARN=$(aws cloudformation describe-stacks \
  --stack-name OpenSearchFineTuneStack \
  --query 'Stacks[0].Outputs[?OutputKey==`StateMachineArn`].OutputValue' \
  --output text)

aws stepfunctions start-execution \
  --state-machine-arn $STATE_MACHINE_ARN \
  --name my-first-finetune-$(date +%s) \
  --input '{
    "input_type": "opensearch",
    "opensearch_endpoint": "https://your-domain.us-east-1.es.amazonaws.com",
    "index_name": "your-corpus-index",
    "text_field": "content",
    "max_documents": "5000",
    "model_name": "my-finetuned-bge-v1",
    "model_type": "dense",
    "base_model_id": "BAAI/bge-base-en-v1.5",
    "embedding_dimension": "768"
  }'
```

**Example with S3 Corpus:**

First, upload your corpus to S3:
```bash
# Create corpus file (JSONL format)
echo '{"id": "1", "text": "Machine learning is a subset of artificial intelligence."}' > corpus.jsonl
echo '{"id": "2", "text": "Deep learning uses neural networks for pattern recognition."}' >> corpus.jsonl

# Upload to S3
BUCKET=$(aws cloudformation describe-stacks \
  --stack-name OpenSearchFineTuneStack \
  --query 'Stacks[0].Outputs[?OutputKey==`DataBucketName`].OutputValue' \
  --output text)

aws s3 cp corpus.jsonl s3://$BUCKET/input-corpus/corpus.jsonl

# Start execution
aws stepfunctions start-execution \
  --state-machine-arn $STATE_MACHINE_ARN \
  --name my-s3-finetune-$(date +%s) \
  --input "{
    \"input_type\": \"s3\",
    \"s3_corpus_path\": \"s3://$BUCKET/input-corpus/corpus.jsonl\",
    \"max_documents\": \"5000\",
    \"model_name\": \"my-finetuned-bge-v1\",
    \"model_type\": \"dense\",
    \"base_model_id\": \"BAAI/bge-base-en-v1.5\",
    \"embedding_dimension\": \"768\",
    \"opensearch_endpoint\": \"https://your-domain.us-east-1.es.amazonaws.com\"
  }"
```

### 8. Monitor Progress

```bash
# Get execution ARN from previous command
EXECUTION_ARN="<execution-arn-from-previous-output>"

# Check status
aws stepfunctions describe-execution --execution-arn $EXECUTION_ARN

# Stream CloudWatch logs
aws logs tail /aws/stepfunctions/FineTuneStateMachine --follow
```

Expected timeline:
- Data extraction: 2-5 minutes
- Bedrock query generation: 5-15 minutes (batch job)
- Training: 10-30 minutes (depends on data size)
- Endpoint deployment: 5-10 minutes
- Total: ~30-60 minutes

### 9. Test Your Model

Once complete, get the model ID from Step Functions output:

```bash
# Get model ID from execution history
aws stepfunctions get-execution-history \
  --execution-arn $EXECUTION_ARN \
  --query 'events[-1].executionSucceededEventDetails.output' \
  --output text | jq -r '.registration_result.Payload.model_id'

# Or query OpenSearch directly
curl -X GET "https://your-domain.us-east-1.es.amazonaws.com/_plugins/_ml/models" \
  --aws-sigv4 "aws:amz:us-east-1:es"

# Test inference
MODEL_ID="<your-model-id>"
curl -X POST "https://your-domain.us-east-1.es.amazonaws.com/_plugins/_ml/models/$MODEL_ID/_predict" \
  -H "Content-Type: application/json" \
  --aws-sigv4 "aws:amz:us-east-1:es" \
  -d '{
    "text_docs": ["what is machine learning?"]
  }'
```

Expected output:
```json
{
  "inference_results": [
    {
      "output": [
        {
          "data_type": "FLOAT32",
          "data": [0.123, -0.456, 0.789, ...]
        }
      ]
    }
  ]
}
```

### 10. Use in OpenSearch Queries

Create a KNN index with your model:

```bash
curl -X PUT "https://your-domain.us-east-1.es.amazonaws.com/my-semantic-index" \
  --aws-sigv4 "aws:amz:us-east-1:es" \
  -H "Content-Type: application/json" \
  -d '{
    "settings": {
      "index.knn": true
    },
    "mappings": {
      "properties": {
        "text": {"type": "text"},
        "embedding": {
          "type": "knn_vector",
          "dimension": 768,
          "method": {
            "name": "hnsw",
            "engine": "nmslib",
            "parameters": {}
          }
        }
      }
    }
  }'

# Index documents with embeddings from your model
# Then search semantically!
```

## Troubleshooting

### Stack Deployment Fails
- Check IAM permissions
- Ensure region supports Bedrock
- Verify OpenSearch endpoint is accessible

### Bedrock Access Denied
```bash
# Request model access in AWS Console
# Bedrock → Model access → Request access for Claude models
```

### Training Fails
```bash
# Check logs
aws logs tail /aws/sagemaker/TrainingJobs --follow

# Common issues:
# - OOM: Reduce batch size in input parameters
# - Wrong model ID: Verify HuggingFace model exists
```

### OpenSearch Registration Fails
```bash
# Verify role mapping (Step 6)
# Check OpenSearch access policies
# Ensure endpoint is correct
```

## Clean Up

```bash
# Delete stack
cdk destroy

# Or via AWS Console
aws cloudformation delete-stack --stack-name OpenSearchFineTuneStack

# Delete SageMaker endpoints manually to stop charges
aws sagemaker list-endpoints
aws sagemaker delete-endpoint --endpoint-name <endpoint-name>
```

## Next Steps

- Read [README.md](README.md) for detailed documentation
- Review [scope.md](../scope.md) for project architecture
- Check [workflow.md](../workflow.md) for workflow diagram
- Explore training parameters for better performance
- Try different base models (BGE, GTE, etc.)

## Support

- GitHub Issues: Report bugs and request features
- AWS Support: For AWS service issues
- OpenSearch Forums: For OpenSearch-specific questions
