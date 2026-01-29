#!/bin/bash
# Quick test script for query generation workflow

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Query Generation Workflow - Quick Test${NC}"
echo -e "${BLUE}========================================${NC}\n"

# Check if required arguments are provided
if [ "$#" -lt 2 ]; then
    echo -e "${RED}Usage: $0 <source-s3-path> <data-bucket> [max-documents] [bedrock-role-arn]${NC}"
    echo ""
    echo "Examples:"
    echo "  # Dry run (prepare data only)"
    echo "  $0 s3://my-bucket/corpus.jsonl my-data-bucket 10"
    echo ""
    echo "  # Full run (create Bedrock job)"
    echo "  $0 s3://my-bucket/corpus.jsonl my-data-bucket 10 arn:aws:iam::123456:role/BedrockRole"
    echo ""
    echo "Note: Source file must be in JSONL format with 'id' and 'text' fields"
    exit 1
fi

SOURCE_PATH=$1
DATA_BUCKET=$2
MAX_DOCS=${3:-10}
BEDROCK_ROLE=${4:-""}

echo -e "${GREEN}Configuration:${NC}"
echo "  Source: $SOURCE_PATH"
echo "  Data Bucket: $DATA_BUCKET"
echo "  Max Documents: $MAX_DOCS"
if [ -n "$BEDROCK_ROLE" ]; then
    echo "  Bedrock Role: $BEDROCK_ROLE"
    echo "  Mode: Full run (with Bedrock job)"
else
    echo "  Mode: Dry run (no Bedrock job)"
fi
echo ""

# Check if data bucket exists
echo -e "${BLUE}Checking data bucket...${NC}"
if aws s3 ls "s3://$DATA_BUCKET" > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Bucket exists${NC}\n"
else
    echo -e "${YELLOW}⚠ Bucket does not exist. Creating...${NC}"
    aws s3 mb "s3://$DATA_BUCKET"
    echo -e "${GREEN}✓ Bucket created${NC}\n"
fi

# Check if source file exists
echo -e "${BLUE}Checking source file...${NC}"
if aws s3 ls "$SOURCE_PATH" > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Source file exists${NC}\n"
else
    echo -e "${RED}✗ Source file not found: $SOURCE_PATH${NC}"
    echo "Please upload your corpus file first (must be JSONL format):"
    echo "  aws s3 cp your-corpus.jsonl $SOURCE_PATH"
    echo ""
    echo "JSONL format example:"
    echo '  {"id": "doc1", "text": "Your document text here"}'
    echo '  {"id": "doc2", "text": "Another document"}'
    exit 1
fi

# Run the test
echo -e "${BLUE}Running test...${NC}\n"

if [ -n "$BEDROCK_ROLE" ]; then
    # Full run
    python test_query_generation.py \
        --source "$SOURCE_PATH" \
        --data-bucket "$DATA_BUCKET" \
        --max-documents "$MAX_DOCS" \
        --role-arn "$BEDROCK_ROLE"
else
    # Dry run
    python test_query_generation.py \
        --source "$SOURCE_PATH" \
        --data-bucket "$DATA_BUCKET" \
        --max-documents "$MAX_DOCS" \
        --dry-run
fi

TEST_RESULT=$?

if [ $TEST_RESULT -eq 0 ]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Test completed successfully!${NC}"
    echo -e "${GREEN}========================================${NC}\n"

    echo -e "${BLUE}Next steps:${NC}"
    echo "1. View processed documents:"
    echo "   aws s3 cp s3://$DATA_BUCKET/raw-corpus/documents.jsonl -"
    echo ""
    echo "2. View Bedrock input:"
    echo "   aws s3 cp s3://$DATA_BUCKET/bedrock-input/batch_input.jsonl -"
    echo ""

    if [ -n "$BEDROCK_ROLE" ]; then
        echo "3. Check Bedrock job status:"
        echo "   aws bedrock get-model-invocation-job --job-identifier <JOB-ID>"
        echo ""
        echo "4. View generated queries (after job completes):"
        echo "   aws s3 ls s3://$DATA_BUCKET/bedrock-output/"
    else
        echo "3. To create Bedrock job, run with role ARN:"
        echo "   $0 $SOURCE_PATH $DATA_BUCKET $MAX_DOCS <bedrock-role-arn>"
    fi
    echo ""
else
    echo ""
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}Test failed!${NC}"
    echo -e "${RED}========================================${NC}"
    exit 1
fi
