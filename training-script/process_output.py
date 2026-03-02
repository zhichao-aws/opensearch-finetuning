"""
Process Bedrock Batch Output and Create Training Data with Hard Negatives

Runs as a SageMaker training job to handle large datasets that would
exceed Lambda's 15-minute timeout.

Processes Bedrock batch inference output, extracts synthetic queries,
mines BM25 hard negatives, and creates training data.

Input (via hyperparameters):
    --output-s3-path: S3 path to Bedrock batch output directory
    --documents-s3-path: S3 path to corpus documents JSONL
    --num-negatives: Number of hard negatives per query (default: 20)

Output: training_data.jsonl written to SM_MODEL_DIR (/opt/ml/model/)

Usage:
    python process_output.py --output-s3-path s3://... --documents-s3-path s3://...
"""

import argparse
import json
import os
import re
import logging
import time
from multiprocessing import Pool
from pathlib import Path
from urllib.parse import urlparse

import boto3
import bm25s
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Shared data for multiprocessing workers (set before Pool creation, inherited via fork)
_shared_retriever = None
_shared_query_tokens = None


def _retrieve_range(args):
    """Worker: retrieve BM25 results for a slice of queries."""
    start, end, top_k = args
    chunk = _shared_query_tokens[start:end]
    results, scores = _shared_retriever.retrieve(chunk, k=top_k)
    return results, scores


def parse_s3_path(s3_path):
    """Parse S3 path into bucket and key."""
    parsed = urlparse(s3_path)
    return parsed.netloc, parsed.path.lstrip('/')


def read_s3_jsonl(s3_path):
    """Read JSONL file from S3."""
    s3 = boto3.client('s3')
    bucket, key = parse_s3_path(s3_path)
    response = s3.get_object(Bucket=bucket, Key=key)
    content = response['Body'].read().decode('utf-8')
    return [json.loads(line) for line in content.strip().split('\n') if line.strip()]


def extract_queries_from_model_output(model_output):
    """Extract queries from Claude model output."""
    try:
        output_content = model_output.get('content', [])
        generated_text = ''
        for block in output_content:
            if block.get('type') == 'text':
                generated_text = block.get('text', '')
                break

        if not generated_text:
            return []

        # Parse queries from JSON response {"queries": [...]}
        text = generated_text.strip()
        if text.startswith('```'):
            text = re.sub(r'^```\w*\n?', '', text)
            text = re.sub(r'\n?```$', '', text)

        parsed = json.loads(text)
        queries = parsed.get('queries', []) if isinstance(parsed, dict) else parsed
        return queries

    except json.JSONDecodeError:
        # Try to find JSON object in text
        match = re.search(r'\{.*\}', generated_text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
                return parsed.get('queries', [])
            except json.JSONDecodeError:
                pass
        return []


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process Bedrock batch output and create training data with hard negatives"
    )

    # SageMaker environment variables
    parser.add_argument("--model-dir", type=str,
                        default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))

    # S3 paths (passed as hyperparameters from Step Functions)
    parser.add_argument("--output-s3-path", type=str, required=True,
                        help="S3 path to Bedrock batch output directory")
    parser.add_argument("--documents-s3-path", type=str, required=True,
                        help="S3 path to corpus documents JSONL")
    parser.add_argument("--num-negatives", type=int, default=20,
                        help="Number of hard negatives per query")
    parser.add_argument("--max-corpus-documents", type=int, default=0,
                        help="Max documents for BM25 corpus (0 = unlimited)")

    return parser.parse_args()


def main():
    args = parse_args()
    s3 = boto3.client('s3')

    # Load corpus documents
    t0 = time.time()
    logger.info(f"Loading documents from {args.documents_s3_path}")
    documents = read_s3_jsonl(args.documents_s3_path)

    doc_ids = []
    doc_texts = []
    doc_id_to_idx = {}

    for idx, doc in enumerate(documents):
        doc_id = doc.get('id', str(idx))
        doc_text = doc['text']
        doc_ids.append(doc_id)
        doc_texts.append(doc_text)
        doc_id_to_idx[doc_id] = idx

    # Apply corpus size limit for BM25 pool
    if args.max_corpus_documents > 0 and len(doc_texts) > args.max_corpus_documents:
        logger.info(f"Truncating corpus from {len(doc_texts)} to {args.max_corpus_documents} documents")
        doc_ids = doc_ids[:args.max_corpus_documents]
        doc_texts = doc_texts[:args.max_corpus_documents]
        doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}

    del documents  # Free raw dicts to save memory
    logger.info(f"Loaded {len(doc_texts)} documents in {time.time() - t0:.1f}s")

    # List Bedrock output files
    output_bucket, output_prefix = parse_s3_path(args.output_s3_path)
    paginator = s3.get_paginator('list_objects_v2')
    output_files = []
    for page in paginator.paginate(Bucket=output_bucket, Prefix=output_prefix):
        for obj in page.get('Contents', []):
            if obj['Key'].endswith('.jsonl.out'):
                output_files.append(obj['Key'])

    logger.info(f"Found {len(output_files)} Bedrock output files")

    # Parse Bedrock output to collect query-document pairs
    logger.info("Parsing Bedrock output to collect query-document pairs...")
    query_doc_pairs = []
    errors = 0

    for output_key in output_files:
        try:
            response = s3.get_object(Bucket=output_bucket, Key=output_key)
            content = response['Body'].read().decode('utf-8')

            for line in content.strip().split('\n'):
                if not line.strip():
                    continue

                try:
                    record = json.loads(line)
                    record_id = record.get('recordId')
                    model_output = record.get('modelOutput', {})

                    if record_id not in doc_id_to_idx:
                        continue

                    queries = extract_queries_from_model_output(model_output)
                    if not queries:
                        errors += 1
                        continue

                    for query in queries:
                        if isinstance(query, str) and query.strip():
                            query_doc_pairs.append({
                                "query": query.strip(),
                                "doc_id": record_id,
                                "doc_text": doc_texts[doc_id_to_idx[record_id]]
                            })

                except Exception as e:
                    logger.warning(f"Error processing record: {e}")
                    errors += 1

        except Exception as e:
            logger.error(f"Error processing file {output_key}: {e}")

    logger.info(f"Collected {len(query_doc_pairs)} query-document pairs, {errors} errors")

    if not query_doc_pairs:
        raise ValueError("No query-document pairs generated")

    # Extract queries and positive doc info for batch processing
    queries = [p["query"] for p in query_doc_pairs]
    positive_doc_ids = [p["doc_id"] for p in query_doc_pairs]
    positive_doc_texts = [p["doc_text"] for p in query_doc_pairs]

    # Mine hard negatives using BM25
    num_negatives = args.num_negatives
    logger.info(f"Mining hard negatives (num_negatives={num_negatives})...")

    # Build BM25 index over corpus
    t0 = time.time()
    logger.info(f"Tokenizing and indexing {len(doc_texts)} documents...")
    corpus_tokens = bm25s.tokenize(doc_texts)
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    del corpus_tokens
    logger.info(f"BM25 index built in {time.time() - t0:.1f}s")

    # Tokenize queries
    top_k = min(num_negatives + 5, len(doc_texts))
    query_tokens = bm25s.tokenize(queries, return_ids=False)

    # Retrieve in parallel across all CPU cores
    global _shared_retriever, _shared_query_tokens
    _shared_retriever = retriever
    _shared_query_tokens = query_tokens

    n_processes = os.cpu_count() or 1
    n_queries = len(queries)
    chunk_size = (n_queries + n_processes - 1) // n_processes
    ranges = [(i, min(i + chunk_size, n_queries), top_k) for i in range(0, n_queries, chunk_size)]

    logger.info(f"Retrieving top-{top_k} for {n_queries} queries with {n_processes} parallel workers...")
    t0 = time.time()
    with Pool(n_processes) as pool:
        chunk_results = pool.map(_retrieve_range, ranges)

    results = np.concatenate([r[0] for r in chunk_results], axis=0)
    scores = np.concatenate([r[1] for r in chunk_results], axis=0)
    del chunk_results
    _shared_retriever = None
    _shared_query_tokens = None
    logger.info(f"Retrieval completed in {time.time() - t0:.1f}s")

    # Build training samples with hard negatives
    training_samples = []
    for i, (query, pos_doc_id, pos_doc_text) in enumerate(
        zip(queries, positive_doc_ids, positive_doc_texts)
    ):
        pos_idx = doc_id_to_idx.get(pos_doc_id, -1)
        hard_negatives = []
        for j in range(results.shape[1]):
            doc_idx = int(results[i, j])
            if doc_idx != pos_idx:
                hard_negatives.append(doc_texts[doc_idx])
                if len(hard_negatives) >= num_negatives:
                    break
        training_samples.append({
            "anchor": query,
            "positive": pos_doc_text,
            "negatives": hard_negatives
        })

        if (i + 1) % 10000 == 0:
            logger.info(f"Processed {i + 1}/{len(queries)} queries")

    logger.info(f"Created {len(training_samples)} training samples with hard negatives")

    # Write output to SageMaker model directory
    output_dir = Path(args.model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "training_data.jsonl"

    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in training_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    logger.info(f"Training data written to {output_path} ({len(training_samples)} samples)")


if __name__ == "__main__":
    main()
