# Changes Summary

## Overview
Updated the S3 data validator to enforce strict JSONL format validation without re-saving files, and updated all related components.

## Changed Files

### 1. Lambda Functions

#### `lambda_functions/s3_data_validator/index.py`
**Before**:
- Accepted multiple formats (plain text, JSONL with or without fields)
- Auto-converted and saved to `raw-corpus/documents.jsonl`
- Lenient validation

**After**:
- **Only accepts JSONL format** with `.jsonl` extension
- **Strict validation**:
  - Both `id` and `text` fields are required
  - Both fields must be strings
  - `text` cannot be empty
- **No re-saving**: Returns original file path
- Throws clear validation errors with line numbers

#### `lambda_functions/bedrock_batch_orchestrator/prompt_templates.py`
**Changed**:
- Query generation count: 2 → 5 queries per document
- Added 5 diverse query types:
  - Factoid queries
  - Conceptual queries
  - How-to queries
  - Comparison queries
  - Problem-solving queries

### 2. CDK Infrastructure

#### `step_functions/state_machine_definition.py`
**Added**:
- Two normalization Pass states:
  - `NormalizeOpenSearchPath`: Extracts path from OpenSearch extractor result
  - `NormalizeS3Path`: Extracts path from S3 validator result
- Both paths converge to a common `documents_s3_path` field

**Updated**:
- Bedrock input preparation now uses normalized path
- Workflow chains updated to go through normalization steps

### 3. Test & Documentation

#### `sample_corpus.txt` → `sample_corpus.jsonl`
**Changed**:
- Converted from plain text to JSONL format
- Added proper `id` and `text` fields for each document
- Renamed file extension to `.jsonl`

#### `test_query_generation.py`
**Updated**:
- Updated test messages to reflect validation-only behavior
- No longer mentions "saving" processed data

#### `TEST_QUERY_GENERATION.md`
**Updated**:
- Added strict format requirements section
- Removed references to plain text support
- Added invalid format examples
- Updated sample commands to use `.jsonl` extension
- Clarified validator doesn't save files

#### `quick_test.sh`
**Updated**:
- Usage examples now use `.jsonl` extension
- Added JSONL format requirements in help text
- Updated error messages to show JSONL example

## Breaking Changes

### Input Format
**Before**: Accepted plain text files, JSONL with optional fields
```text
This is a document.
Another document here.
```

**After**: Only accepts strict JSONL with required fields
```json
{"id": "doc1", "text": "This is a document."}
{"id": "doc2", "text": "Another document here."}
```

### Validation Behavior
**Before**:
- Lenient parsing
- Auto-generated IDs
- Always saved to `s3://DATA-BUCKET/raw-corpus/documents.jsonl`

**After**:
- Strict validation
- IDs must be provided
- Returns original file path, no saving

## Migration Guide

### For Existing Users

1. **Convert your corpus files to JSONL**:
```bash
# From plain text
python << 'EOF'
import json
with open('corpus.txt') as f:
    with open('corpus.jsonl', 'w') as out:
        for i, line in enumerate(f):
            doc = {"id": f"doc{i}", "text": line.strip()}
            out.write(json.dumps(doc) + '\n')
EOF
```

2. **Validate your JSONL file**:
```python
import json

with open('corpus.jsonl') as f:
    for i, line in enumerate(f, 1):
        doc = json.loads(line)
        assert 'id' in doc, f"Missing 'id' on line {i}"
        assert 'text' in doc, f"Missing 'text' on line {i}"
        assert isinstance(doc['id'], str), f"'id' must be string on line {i}"
        assert isinstance(doc['text'], str), f"'text' must be string on line {i}"
        assert doc['text'].strip(), f"'text' cannot be empty on line {i}"
```

3. **Update S3 paths**:
```bash
# Upload with .jsonl extension
aws s3 cp corpus.jsonl s3://YOUR-BUCKET/corpus/data.jsonl

# Update your workflow inputs to point to .jsonl files
```

## Benefits

1. **Data Quality**: Strict validation ensures consistent, high-quality training data
2. **Performance**: No unnecessary file copying for S3 inputs
3. **Clarity**: Clear error messages help identify data issues early
4. **Efficiency**: 2.5x more training pairs with 5 queries per document
5. **Consistency**: Both S3 and OpenSearch paths handled uniformly

## Backward Compatibility

⚠️ **This is a breaking change**. Existing plain text corpus files will no longer work and must be converted to JSONL format.

## Testing

To test with the new format:
```bash
# Use the sample file
./quick_test.sh \
  s3://YOUR-BUCKET/corpus/sample_corpus.jsonl \
  YOUR-DATA-BUCKET \
  10
```

## Future Enhancements

Potential improvements:
- Add support for additional optional fields (metadata, embeddings)
- Batch validation for large files
- Streaming validation for memory efficiency
- Schema versioning for evolving requirements
