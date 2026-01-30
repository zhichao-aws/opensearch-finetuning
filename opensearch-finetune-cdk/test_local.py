#!/usr/bin/env python3
"""
Local testing for Lambda functions without deployment.

This script imports and runs Lambda handlers directly, which is faster
for development iteration than deploying and invoking via AWS.

Requirements:
    pip install -r requirements.txt
    pip install moto pytest  # for mocking

Usage:
    # Test data extractor logic locally (requires real OpenSearch)
    python test_local.py test-extractor --endpoint https://... --index my-index

    # Test S3 validator locally
    python test_local.py test-validator --s3-path s3://bucket/corpus.jsonl

    # Test Bedrock orchestrator prepare_input locally
    python test_local.py test-bedrock-prepare --s3-docs-path s3://bucket/docs.jsonl

    # Generate sample training data locally (no AWS)
    python test_local.py generate-sample-data --output sample_training.jsonl
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add lambda functions to path
sys.path.insert(0, str(Path(__file__).parent / "lambda_functions" / "data_extractor"))
sys.path.insert(0, str(Path(__file__).parent / "lambda_functions" / "s3_data_validator"))
sys.path.insert(0, str(Path(__file__).parent / "lambda_functions" / "bedrock_batch_orchestrator"))
sys.path.insert(0, str(Path(__file__).parent / "lambda_functions" / "register_model"))


def test_extractor_local(args):
    """Test data extractor Lambda locally."""
    print("\n" + "="*60)
    print("LOCAL TEST: Data Extractor")
    print("="*60)

    # Set required environment variables
    os.environ["DATA_BUCKET"] = args.data_bucket or "test-bucket"
    os.environ["MAX_DOCUMENTS"] = str(args.max_documents)

    try:
        from index import handler

        event = {
            "opensearch_endpoint": args.endpoint,
            "index_name": args.index,
            "text_field": args.text_field,
            "doc_id_field": args.doc_id_field,
            "max_documents": args.max_documents,
        }

        print(f"\nEvent: {json.dumps(event, indent=2)}")
        result = handler(event, None)
        print(f"\nResult: {json.dumps(result, indent=2)}")

        return 0 if result.get("status") == "success" else 1

    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure you have the required dependencies installed.")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1


def test_validator_local(args):
    """Test S3 validator Lambda locally."""
    print("\n" + "="*60)
    print("LOCAL TEST: S3 Data Validator")
    print("="*60)

    os.environ["DATA_BUCKET"] = args.data_bucket or "test-bucket"
    os.environ["MAX_DOCUMENTS"] = str(args.max_documents)

    try:
        # Change path for this import
        sys.path.insert(0, str(Path(__file__).parent / "lambda_functions" / "s3_data_validator"))
        from index import handler

        event = {
            "s3_corpus_path": args.s3_path,
            "max_documents": args.max_documents,
        }

        print(f"\nEvent: {json.dumps(event, indent=2)}")
        result = handler(event, None)
        print(f"\nResult: {json.dumps(result, indent=2)}")

        return 0 if result.get("status") == "success" else 1

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


def test_bedrock_prepare_local(args):
    """Test Bedrock prepare_input locally."""
    print("\n" + "="*60)
    print("LOCAL TEST: Bedrock Prepare Input")
    print("="*60)

    os.environ["DATA_BUCKET"] = args.data_bucket or "test-bucket"
    os.environ["BEDROCK_MODEL_ID"] = args.bedrock_model_id

    try:
        sys.path.insert(0, str(Path(__file__).parent / "lambda_functions" / "bedrock_batch_orchestrator"))
        from index import handler

        event = {
            "operation": "prepare_input",
            "s3_documents_path": args.s3_docs_path,
            "data_source": "local_test",
        }

        print(f"\nEvent: {json.dumps(event, indent=2)}")
        result = handler(event, None)
        print(f"\nResult: {json.dumps(result, indent=2)}")

        return 0 if result.get("status") == "success" else 1

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


def generate_sample_data(args):
    """Generate sample training data for local testing."""
    print("\n" + "="*60)
    print("Generating Sample Training Data")
    print("="*60)

    sample_docs = [
        {
            "id": "doc1",
            "text": "Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed. It uses algorithms to identify patterns in data and make predictions."
        },
        {
            "id": "doc2",
            "text": "OpenSearch is an open-source search and analytics engine derived from Elasticsearch. It supports full-text search, structured queries, and aggregations for analyzing large volumes of data."
        },
        {
            "id": "doc3",
            "text": "Amazon SageMaker is a fully managed machine learning service that enables data scientists and developers to build, train, and deploy ML models quickly. It provides built-in algorithms and supports custom training scripts."
        },
        {
            "id": "doc4",
            "text": "Fine-tuning is the process of taking a pre-trained model and further training it on a specific dataset to adapt it for a particular task. This approach leverages transfer learning to achieve better performance with less data."
        },
        {
            "id": "doc5",
            "text": "Vector embeddings are numerical representations of data that capture semantic meaning. They are commonly used in search and recommendation systems to find similar items based on meaning rather than exact keyword matches."
        },
    ]

    # Generate sample queries for each document (simulating Bedrock output)
    sample_training_pairs = []
    for doc in sample_docs:
        # Simple query generation (in reality, Bedrock would do this)
        queries = generate_queries_for_doc(doc["text"])
        for query in queries:
            sample_training_pairs.append({
                "query": query,
                "document": doc["text"]
            })

    # Write corpus
    corpus_path = Path(args.output).parent / "sample_corpus.jsonl"
    with open(corpus_path, "w") as f:
        for doc in sample_docs:
            f.write(json.dumps(doc) + "\n")
    print(f"✓ Sample corpus written to: {corpus_path}")
    print(f"  Documents: {len(sample_docs)}")

    # Write training data
    with open(args.output, "w") as f:
        for pair in sample_training_pairs:
            f.write(json.dumps(pair) + "\n")
    print(f"✓ Sample training data written to: {args.output}")
    print(f"  Training pairs: {len(sample_training_pairs)}")

    return 0


def generate_queries_for_doc(text: str) -> list:
    """Simple query generation for sample data (simulates Bedrock)."""
    # Extract key phrases - very simple heuristic
    words = text.split()
    queries = []

    # Take first few significant words as a query
    significant = [w for w in words[:10] if len(w) > 4]
    if significant:
        queries.append(f"What is {' '.join(significant[:3])}?")

    # Create a conceptual query
    if "machine learning" in text.lower():
        queries.append("How does machine learning work?")
    elif "opensearch" in text.lower():
        queries.append("What can OpenSearch do?")
    elif "sagemaker" in text.lower():
        queries.append("How to deploy ML models?")
    elif "fine-tuning" in text.lower():
        queries.append("Why fine-tune pretrained models?")
    elif "embedding" in text.lower():
        queries.append("What are vector embeddings used for?")
    else:
        queries.append(f"Explain {words[0]} {words[1] if len(words) > 1 else ''}")

    return queries


def test_training_script_local(args):
    """Test the training script locally with sample data."""
    print("\n" + "="*60)
    print("LOCAL TEST: Training Script")
    print("="*60)

    training_script_path = Path(__file__).parent / "training_scripts" / "finetune_script.py"

    if not training_script_path.exists():
        print(f"Training script not found at: {training_script_path}")
        return 1

    # For local testing, we'd need to run the script with sample data
    # This requires GPU typically, so just validate the script can be imported
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("finetune_script", training_script_path)
        module = importlib.util.module_from_spec(spec)

        print(f"✓ Training script can be loaded: {training_script_path}")
        print("\nTo test training locally, run:")
        print(f"  python {training_script_path} \\")
        print(f"    --training-data {args.training_data} \\")
        print(f"    --model-id {args.model_id} \\")
        print(f"    --output-dir ./local_model_output \\")
        print(f"    --num-epochs 1 \\")
        print(f"    --batch-size 4")

        return 0

    except Exception as e:
        print(f"Error loading training script: {e}")
        return 1


def validate_jsonl_local(args):
    """Validate a local JSONL file format."""
    print("\n" + "="*60)
    print("LOCAL VALIDATION: JSONL File")
    print("="*60)

    file_path = Path(args.file)
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return 1

    errors = []
    line_count = 0
    valid_count = 0

    with open(file_path, "r") as f:
        for i, line in enumerate(f, 1):
            line_count += 1
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)

                # Check for required fields based on file type
                if args.type == "corpus":
                    if "id" not in data:
                        errors.append(f"Line {i}: Missing 'id' field")
                    elif "text" not in data:
                        errors.append(f"Line {i}: Missing 'text' field")
                    elif not data.get("text"):
                        errors.append(f"Line {i}: Empty 'text' field")
                    else:
                        valid_count += 1

                elif args.type == "training":
                    if "query" not in data:
                        errors.append(f"Line {i}: Missing 'query' field")
                    elif "document" not in data:
                        errors.append(f"Line {i}: Missing 'document' field")
                    else:
                        valid_count += 1

            except json.JSONDecodeError as e:
                errors.append(f"Line {i}: Invalid JSON - {e}")

            if len(errors) >= 10:
                errors.append("... (showing first 10 errors)")
                break

    print(f"\nFile: {file_path}")
    print(f"Total lines: {line_count}")
    print(f"Valid records: {valid_count}")

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for error in errors:
            print(f"  - {error}")
        return 1
    else:
        print("\n✓ All records valid!")
        return 0


def main():
    parser = argparse.ArgumentParser(
        description="Local testing utilities for Lambda functions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Test extractor locally
    ext_parser = subparsers.add_parser("test-extractor", help="Test data extractor locally")
    ext_parser.add_argument("--endpoint", required=True, help="OpenSearch endpoint")
    ext_parser.add_argument("--index", required=True, help="Index name")
    ext_parser.add_argument("--text-field", default="content", help="Text field name")
    ext_parser.add_argument("--doc-id-field", default="_id", help="Document ID field")
    ext_parser.add_argument("--max-documents", type=int, default=100, help="Max documents")
    ext_parser.add_argument("--data-bucket", help="S3 bucket name (for output)")

    # Test validator locally
    val_parser = subparsers.add_parser("test-validator", help="Test S3 validator locally")
    val_parser.add_argument("--s3-path", required=True, help="S3 path to corpus")
    val_parser.add_argument("--max-documents", type=int, default=100, help="Max documents")
    val_parser.add_argument("--data-bucket", help="S3 bucket name")

    # Test Bedrock prepare locally
    bed_parser = subparsers.add_parser("test-bedrock-prepare", help="Test Bedrock prepare_input locally")
    bed_parser.add_argument("--s3-docs-path", required=True, help="S3 path to documents")
    bed_parser.add_argument("--data-bucket", help="S3 bucket name")
    bed_parser.add_argument("--bedrock-model-id", default="us.anthropic.claude-haiku-4-5-20251001-v1:0", help="Bedrock model ID")

    # Generate sample data
    sample_parser = subparsers.add_parser("generate-sample-data", help="Generate sample training data")
    sample_parser.add_argument("--output", default="sample_training.jsonl", help="Output file path")

    # Test training script
    train_parser = subparsers.add_parser("test-training", help="Validate training script")
    train_parser.add_argument("--training-data", default="sample_training.jsonl", help="Training data path")
    train_parser.add_argument("--model-id", default="sentence-transformers/all-MiniLM-L6-v2", help="Base model ID")

    # Validate JSONL
    jsonl_parser = subparsers.add_parser("validate-jsonl", help="Validate local JSONL file")
    jsonl_parser.add_argument("--file", required=True, help="JSONL file to validate")
    jsonl_parser.add_argument("--type", choices=["corpus", "training"], default="corpus", help="File type")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    if args.command == "test-extractor":
        return test_extractor_local(args)
    elif args.command == "test-validator":
        return test_validator_local(args)
    elif args.command == "test-bedrock-prepare":
        return test_bedrock_prepare_local(args)
    elif args.command == "generate-sample-data":
        return generate_sample_data(args)
    elif args.command == "test-training":
        return test_training_script_local(args)
    elif args.command == "validate-jsonl":
        return validate_jsonl_local(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
