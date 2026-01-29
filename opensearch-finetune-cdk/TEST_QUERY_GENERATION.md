# 测试查询生成工作流

本指南说明如何单独测试从 S3 读取数据并生成 sample query 的部分，无需部署完整的 fine-tuning pipeline。

## 方案 1: 本地测试脚本（推荐，最快）

使用 `test_query_generation.py` 脚本直接调用 Lambda 函数逻辑。

### 准备工作

1. 准备测试数据（必须是 JSONL 格式），上传到 S3:
```bash
# 创建示例语料库文件 (JSONL format)
cat > sample_corpus.jsonl << 'EOF'
{"id": "doc1", "text": "Amazon Web Services (AWS) is a comprehensive cloud computing platform."}
{"id": "doc2", "text": "Machine learning enables computers to learn from data without explicit programming."}
{"id": "doc3", "text": "Python is a popular programming language for data science and AI."}
{"id": "doc4", "text": "OpenSearch is an open-source search and analytics engine."}
{"id": "doc5", "text": "Fine-tuning adapts pre-trained models to specific tasks."}
EOF

# 上传到 S3
aws s3 cp sample_corpus.jsonl s3://YOUR-BUCKET/corpus/sample_corpus.jsonl
```

**重要**:
- 文件必须是 `.jsonl` 扩展名
- 每行必须是有效的 JSON，包含 `id` 和 `text` 两个字段
- `id` 和 `text` 都必须是字符串类型
- `text` 不能为空

2. 创建数据桶（如果还没有）:
```bash
aws s3 mb s3://YOUR-DATA-BUCKET
```

### 运行测试

**Dry run 模式** (只准备数据，不创建 Bedrock job):
```bash
python test_query_generation.py \
  --source s3://YOUR-BUCKET/corpus/sample_corpus.txt \
  --data-bucket YOUR-DATA-BUCKET \
  --max-documents 10 \
  --dry-run
```

**完整运行** (包括创建 Bedrock batch job):

首先需要一个有 Bedrock 权限的 IAM role:
```bash
# 如果没有，可以使用方案 2 部署测试栈来创建
python test_query_generation.py \
  --source s3://YOUR-BUCKET/corpus/sample_corpus.txt \
  --data-bucket YOUR-DATA-BUCKET \
  --role-arn arn:aws:iam::ACCOUNT:role/BedrockBatchRole \
  --max-documents 10
```

### 输出示例

```
============================================================
Testing Query Generation Workflow
============================================================

Step 1: Validating corpus from s3://my-bucket/corpus.txt
  - Max documents: 10
  ✓ Processed 5 documents
  ✓ Saved to: s3://my-data-bucket/raw-corpus/documents.jsonl

Step 2: Preparing Bedrock batch input
  ✓ Created 5 Bedrock input records
  ✓ Saved to: s3://my-data-bucket/bedrock-input/batch_input.jsonl

Step 3: Creating Bedrock batch job
  ✓ Created Bedrock job: abc123def456
  ✓ Job ARN: arn:aws:bedrock:...
  ✓ Output will be saved to: s3://my-data-bucket/bedrock-output/

============================================================
✅ Test completed successfully!
============================================================
```

### 查看生成的文件

```bash
# 查看处理后的文档
aws s3 cp s3://YOUR-DATA-BUCKET/raw-corpus/documents.jsonl - | head -5

# 查看 Bedrock 输入
aws s3 cp s3://YOUR-DATA-BUCKET/bedrock-input/batch_input.jsonl - | head -5

# 检查 Bedrock job 状态
aws bedrock get-model-invocation-job --job-identifier JOB-ID

# 查看生成的查询 (job 完成后)
aws s3 ls s3://YOUR-DATA-BUCKET/bedrock-output/
```

---

## 方案 2: 部署最小测试栈

部署一个简化的 CDK 栈，只包含测试所需的 Lambda 和 IAM 资源。

### 1. 部署测试栈

```bash
# 修复 bootstrap 问题（如果需要）
aws cloudformation continue-update-rollback --stack-name CDKToolkit --region us-east-1
# 或者删除重建
# aws cloudformation delete-stack --stack-name CDKToolkit --region us-east-1
# cdk bootstrap

# 部署测试栈
cdk deploy TestQueryGenStack -a "python test_query_gen_stack.py"
```

### 2. 获取输出值

```bash
# 获取 stack outputs
aws cloudformation describe-stacks \
  --stack-name TestQueryGenStack \
  --query 'Stacks[0].Outputs' \
  --output table
```

记录以下值:
- `DataBucketName`: 数据桶名称
- `S3ValidatorLambdaArn`: S3 验证器 Lambda ARN
- `BedrockOrchestratorLambdaArn`: Bedrock 编排器 Lambda ARN
- `BedrockBatchRoleArn`: Bedrock Batch IAM Role ARN

### 3. 使用 AWS CLI 调用 Lambda

**步骤 1: 验证 S3 数据**
```bash
aws lambda invoke \
  --function-name $(aws cloudformation describe-stacks \
    --stack-name TestQueryGenStack \
    --query 'Stacks[0].Outputs[?OutputKey==`S3ValidatorLambdaArn`].OutputValue' \
    --output text) \
  --payload '{"s3_corpus_path": "s3://YOUR-BUCKET/corpus.txt", "max_documents": 10}' \
  --cli-binary-format raw-in-base64-out \
  response1.json

cat response1.json | jq
```

**步骤 2: 准备 Bedrock 输入**
```bash
# 从 response1.json 获取 data_bucket
DATA_BUCKET=$(cat response1.json | jq -r '.data_bucket')

aws lambda invoke \
  --function-name $(aws cloudformation describe-stacks \
    --stack-name TestQueryGenStack \
    --query 'Stacks[0].Outputs[?OutputKey==`BedrockOrchestratorLambdaArn`].OutputValue' \
    --output text) \
  --payload "{\"operation\": \"prepare_input\", \"s3_documents_path\": \"s3://${DATA_BUCKET}/raw-corpus/documents.jsonl\"}" \
  --cli-binary-format raw-in-base64-out \
  response2.json

cat response2.json | jq
```

**步骤 3: 创建 Bedrock Batch Job**
```bash
S3_INPUT_PATH=$(cat response2.json | jq -r '.s3_input_path')
BEDROCK_ROLE=$(aws cloudformation describe-stacks \
  --stack-name TestQueryGenStack \
  --query 'Stacks[0].Outputs[?OutputKey==`BedrockBatchRoleArn`].OutputValue' \
  --output text)

aws lambda invoke \
  --function-name $(aws cloudformation describe-stacks \
    --stack-name TestQueryGenStack \
    --query 'Stacks[0].Outputs[?OutputKey==`BedrockOrchestratorLambdaArn`].OutputValue' \
    --output text) \
  --payload "{\"operation\": \"create_job\", \"s3_input_path\": \"${S3_INPUT_PATH}\", \"bedrock_batch_role_arn\": \"${BEDROCK_ROLE}\"}" \
  --cli-binary-format raw-in-base64-out \
  response3.json

cat response3.json | jq
```

**步骤 4: 检查 Job 状态**
```bash
JOB_ID=$(cat response3.json | jq -r '.job_id')

aws lambda invoke \
  --function-name $(aws cloudformation describe-stacks \
    --stack-name TestQueryGenStack \
    --query 'Stacks[0].Outputs[?OutputKey==`BedrockOrchestratorLambdaArn`].OutputValue' \
    --output text) \
  --payload "{\"operation\": \"check_status\", \"job_id\": \"${JOB_ID}\"}" \
  --cli-binary-format raw-in-base64-out \
  response4.json

cat response4.json | jq
```

### 4. 清理资源

```bash
cdk destroy TestQueryGenStack -a "python test_query_gen_stack.py"
```

---

## 支持的数据格式

**S3 输入数据格式要求（严格）**:

只支持 JSONL 格式，文件必须满足以下要求:

1. **文件扩展名**: 必须是 `.jsonl`
2. **JSON 格式**: 每行必须是有效的 JSON 对象
3. **必需字段**:
   - `id` (string): 文档唯一标识符
   - `text` (string): 文档内容
4. **字段验证**:
   - 两个字段都必须是字符串类型
   - `text` 不能为空或只包含空白字符
   - `id` 必须唯一

### 有效示例:
```json
{"id": "doc1", "text": "This is document 1."}
{"id": "doc2", "text": "This is document 2."}
{"id": "doc3", "text": "This is document 3."}
```

### 无效示例（会抛出异常）:
```json
// ❌ 缺少 id 字段
{"text": "This is document 1."}

// ❌ 缺少 text 字段
{"id": "doc1"}

// ❌ text 为空
{"id": "doc1", "text": ""}

// ❌ id 不是字符串
{"id": 1, "text": "This is document 1."}

// ❌ 不是 JSON 格式
This is plain text.
```

**OpenSearch 输入**: OpenSearch 数据提取器会自动将提取的数据转换为正确的 JSONL 格式并保存到 S3。

---

## 生成的输出格式

### S3 验证器输出
S3 验证器不会创建新文件，只验证现有文件格式并返回原始路径:
```json
{
  "status": "success",
  "s3_path": "s3://YOUR-BUCKET/corpus/sample_corpus.jsonl",  // 原始文件路径
  "document_count": 100
}
```

### OpenSearch 提取器输出 (raw-corpus/documents.jsonl)
OpenSearch 提取器会将数据保存为标准 JSONL 格式:
```json
{"id": "0", "text": "Amazon Web Services (AWS) is..."}
{"id": "1", "text": "Machine learning enables..."}
```

### bedrock-input/batch_input.jsonl
Bedrock Batch API 输入格式:
```json
{
  "recordId": "0",
  "modelInput": {
    "anthropic_version": "bedrock-2023-05-31",
    "messages": [{"role": "user", "content": "Generate queries..."}],
    "max_tokens": 500,
    "temperature": 0.7
  }
}
```

### bedrock-output/*.jsonl.out
Bedrock 输出 (job 完成后):
```json
{
  "recordId": "0",
  "modelOutput": {
    "content": [{"text": "{\"queries\": [\"What is AWS?\", \"Tell me about cloud computing\"]}"}]
  }
}
```

### training-data/training_data.jsonl
最终的训练数据:
```json
{"query": "What is AWS?", "document": "Amazon Web Services (AWS) is..."}
{"query": "Tell me about cloud computing", "document": "Amazon Web Services (AWS) is..."}
```

---

## 故障排查

### 权限错误
确保你的 AWS credentials 有以下权限:
- S3: `s3:GetObject`, `s3:PutObject`, `s3:ListBucket`
- Bedrock: `bedrock:CreateModelInvocationJob`, `bedrock:GetModelInvocationJob`
- Lambda: `lambda:InvokeFunction` (如果使用方案 2)

### Bedrock 模型访问
确保你的账户有 Claude 模型的访问权限:
```bash
aws bedrock list-foundation-models --region us-east-1 | grep claude
```

如果没有，去 AWS Console → Bedrock → Model access 申请。

### Bootstrap 问题
如果 `cdk bootstrap` 失败，使用以下命令修复:
```bash
aws cloudformation continue-update-rollback --stack-name CDKToolkit --region us-east-1
```

---

## 下一步

测试成功后，你可以:
1. 使用更大的数据集 (`--max-documents 1000`)
2. 查看生成的查询质量
3. 调整 prompt templates ([lambda_functions/bedrock_batch_orchestrator/prompt_templates.py](lambda_functions/bedrock_batch_orchestrator/prompt_templates.py))
4. 继续完整的 fine-tuning pipeline
