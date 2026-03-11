"""
Process Bedrock Batch Output and Create Training Data with Hard Negatives

Runs as a SageMaker training job to handle large datasets that would
exceed Lambda's 15-minute timeout.

Processes Bedrock batch inference output, extracts synthetic queries,
mines BM25 hard negatives, and creates training data.

Splits documents into train (80%) and dev (20%) at the document level.
- Train docs: all queries, BM25 negatives for cross-encoder scoring
- Dev docs (capped at 500): 1 query per doc, top-100 BM25 candidates
- Dev set formatted as Bedrock batch input for LLM relevance labeling

Input (via hyperparameters):
    --output-s3-path: S3 path to Bedrock batch output directory
    --documents-s3-path: S3 path to corpus documents JSONL
    --num-negatives: Number of hard negatives per query (default: 20)
    --dev-split-ratio: Fraction of documents for dev set (default: 0.2)
    --max-dev-documents: Max dev documents (default: 500)
    --dev-bm25-top-k: Top-K BM25 candidates for dev queries (default: 100)

Output:
    training_data.jsonl - train set with hard negatives
    dev_data.jsonl - dev set with top-100 BM25 candidates
    dev_bedrock_input.jsonl - Bedrock batch input for LLM scoring dev set
    dev_meta.json - metadata about the dev set
"""

import argparse
import json
import os
import re
import logging
import random
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

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Shared data for multiprocessing workers (set before Pool creation, inherited via fork)
_shared_retriever = None
_shared_query_tokens = None


# LLM relevance scoring prompt (same as llm_label.py)
DEV_LABEL_SYSTEM_PROMPT = """You are an expert search relevance rater. Your task is to evaluate the relevance between search query and results with these criteria:
- Score 1.0: Perfect match, highly relevant
- Score 0.7-0.9: Very relevant with minor variations
- Score 0.4-0.6: Moderately relevant
- Score 0.1-0.3: Slightly relevant
- Score 0.0: Completely irrelevant

Evaluate based on: exact matches, semantic relevance, and overall context between the SearchText and content in Hits.

IMPORTANT: You MUST include a rating for EVERY hit provided.

Return ONLY a JSON object in this EXACT format:
{"ratings": [{"id": "doc_id_here", "rating_score": <score>}]}
Do not include any explanation, commentary, or markdown formatting. Return only the JSON object."""


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


def bm25_retrieve_parallel(retriever, query_tokens, top_k):
    """Retrieve BM25 results using multiprocessing."""
    global _shared_retriever, _shared_query_tokens
    _shared_retriever = retriever
    _shared_query_tokens = query_tokens

    n_processes = os.cpu_count() or 1
    n_queries = len(query_tokens)
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
    return results, scores


def format_dev_bedrock_record(record_id, query, doc_id, doc_text):
    """Format a single (query, doc) pair as a Bedrock batch input record."""
    hits_json = json.dumps([{"_id": doc_id, "_source": {"content": doc_text}}])
    user_content = f"SearchText - {query}; Hits - {hits_json}"

    return {
        "recordId": record_id,
        "modelInput": {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 256,
            "system": DEV_LABEL_SYSTEM_PROMPT,
            "messages": [
                {
                    "role": "user",
                    "content": user_content
                }
            ]
        }
    }


def upload_to_s3(s3_client, bucket, key, data_bytes):
    """Upload bytes to S3."""
    s3_client.put_object(Bucket=bucket, Key=key, Body=data_bytes, ContentType='application/jsonl')
    return f"s3://{bucket}/{key}"


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

    # Dev set configuration
    parser.add_argument("--dev-split-ratio", type=float, default=0.2,
                        help="Fraction of documents for dev set")
    parser.add_argument("--max-dev-documents", type=int, default=1000,
                        help="Max dev documents (1000 x top-50 = 50k Bedrock batch limit)")
    parser.add_argument("--dev-bm25-top-k", type=int, default=50,
                        help="Top-K BM25 candidates for dev queries")
    parser.add_argument("--data-bucket", type=str, default="",
                        help="S3 bucket name for uploading dev Bedrock input")
    parser.add_argument("--model-name-prefix", type=str, default="",
                        help="Model name prefix for namespacing S3 paths")

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

    # Parse Bedrock output to collect query-document pairs, grouped by doc_id
    logger.info("Parsing Bedrock output to collect query-document pairs...")
    doc_queries = {}  # doc_id -> list of queries
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

                    valid_queries = [q.strip() for q in queries if isinstance(q, str) and q.strip()]
                    if valid_queries:
                        doc_queries[record_id] = valid_queries

                except Exception as e:
                    logger.warning(f"Error processing record: {e}")
                    errors += 1

        except Exception as e:
            logger.error(f"Error processing file {output_key}: {e}")

    total_pairs = sum(len(qs) for qs in doc_queries.values())
    logger.info(f"Collected queries for {len(doc_queries)} documents ({total_pairs} total pairs), {errors} errors")

    if not doc_queries:
        raise ValueError("No query-document pairs generated")

    # =========================================================================
    # Split documents into train/dev at the document level
    # =========================================================================
    all_doc_ids_with_queries = sorted(doc_queries.keys())
    random.shuffle(all_doc_ids_with_queries)

    n_dev = min(
        int(len(all_doc_ids_with_queries) * args.dev_split_ratio),
        args.max_dev_documents
    )
    dev_doc_ids = set(all_doc_ids_with_queries[:n_dev])
    train_doc_ids = set(all_doc_ids_with_queries[n_dev:])

    logger.info(f"Split: {len(train_doc_ids)} train docs, {len(dev_doc_ids)} dev docs")

    # =========================================================================
    # Build BM25 index over FULL corpus (shared by train and dev)
    # =========================================================================
    t0 = time.time()
    logger.info(f"Tokenizing and indexing {len(doc_texts)} documents...")
    corpus_tokens = bm25s.tokenize(doc_texts)
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    del corpus_tokens
    logger.info(f"BM25 index built in {time.time() - t0:.1f}s")

    # =========================================================================
    # Process TRAIN set: all queries per doc, mine hard negatives
    # =========================================================================
    logger.info("Processing train set...")
    train_queries = []
    train_positive_doc_ids = []
    train_positive_doc_texts = []

    for did in train_doc_ids:
        for q in doc_queries[did]:
            train_queries.append(q)
            train_positive_doc_ids.append(did)
            train_positive_doc_texts.append(doc_texts[doc_id_to_idx[did]])

    logger.info(f"Train: {len(train_queries)} query-document pairs from {len(train_doc_ids)} docs")

    num_negatives = args.num_negatives
    top_k_train = min(num_negatives + 5, len(doc_texts))
    train_query_tokens = bm25s.tokenize(train_queries, return_ids=False)
    train_results, _ = bm25_retrieve_parallel(retriever, train_query_tokens, top_k_train)

    training_samples = []
    for i, (query, pos_doc_id, pos_doc_text) in enumerate(
        zip(train_queries, train_positive_doc_ids, train_positive_doc_texts)
    ):
        pos_idx = doc_id_to_idx.get(pos_doc_id, -1)
        hard_negatives = []
        for j in range(train_results.shape[1]):
            doc_idx = int(train_results[i, j])
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
            logger.info(f"Processed {i + 1}/{len(train_queries)} train queries")

    logger.info(f"Created {len(training_samples)} training samples with hard negatives")
    del train_results

    # =========================================================================
    # Process DEV set: 1 query per doc, mine top-100 BM25 candidates
    # =========================================================================
    logger.info("Processing dev set...")
    dev_queries = []
    dev_positive_doc_ids = []

    for did in sorted(dev_doc_ids):
        queries_for_doc = doc_queries[did]
        # Sample 1 query per dev document
        selected_query = random.choice(queries_for_doc)
        dev_queries.append(selected_query)
        dev_positive_doc_ids.append(did)

    logger.info(f"Dev: {len(dev_queries)} queries from {len(dev_doc_ids)} docs")

    dev_top_k = min(args.dev_bm25_top_k, len(doc_texts))
    dev_query_tokens = bm25s.tokenize(dev_queries, return_ids=False)
    dev_results, _ = bm25_retrieve_parallel(retriever, dev_query_tokens, dev_top_k)

    dev_samples = []
    dev_bedrock_records = []
    bedrock_record_idx = 0

    for i, (query, pos_doc_id) in enumerate(zip(dev_queries, dev_positive_doc_ids)):
        pos_idx = doc_id_to_idx.get(pos_doc_id, -1)
        candidates = []
        candidate_ids_seen = set()

        # Always include the positive document as first candidate
        candidates.append({
            "id": pos_doc_id,
            "text": doc_texts[pos_idx]
        })
        candidate_ids_seen.add(pos_doc_id)

        # Add BM25 candidates (excluding positive)
        for j in range(dev_results.shape[1]):
            doc_idx = int(dev_results[i, j])
            cand_doc_id = doc_ids[doc_idx]
            if cand_doc_id not in candidate_ids_seen:
                candidates.append({
                    "id": cand_doc_id,
                    "text": doc_texts[doc_idx]
                })
                candidate_ids_seen.add(cand_doc_id)
                if len(candidates) >= dev_top_k:
                    break

        dev_samples.append({
            "query": query,
            "positive_id": pos_doc_id,
            "candidates": candidates
        })

        # Format Bedrock batch records for each (query, candidate) pair
        for cand in candidates:
            record_id = f"dev_{bedrock_record_idx}"
            dev_bedrock_records.append(format_dev_bedrock_record(
                record_id=record_id,
                query=query,
                doc_id=cand["id"],
                doc_text=cand["text"]
            ))
            bedrock_record_idx += 1

    logger.info(f"Created {len(dev_samples)} dev samples with {len(dev_bedrock_records)} total (query, doc) pairs for LLM labeling")
    del dev_results

    # =========================================================================
    # Write outputs
    # =========================================================================
    output_dir = Path(args.model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Training data
    train_path = output_dir / "training_data.jsonl"
    with open(train_path, 'w', encoding='utf-8') as f:
        for sample in training_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    logger.info(f"Training data written to {train_path} ({len(training_samples)} samples)")

    # 2. Dev data (queries + candidates with texts)
    dev_path = output_dir / "dev_data.jsonl"
    with open(dev_path, 'w', encoding='utf-8') as f:
        for sample in dev_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    logger.info(f"Dev data written to {dev_path} ({len(dev_samples)} samples)")

    # 3. Dev Bedrock batch input for LLM labeling
    dev_bedrock_path = output_dir / "dev_bedrock_input.jsonl"
    with open(dev_bedrock_path, 'w', encoding='utf-8') as f:
        for record in dev_bedrock_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    logger.info(f"Dev Bedrock input written to {dev_bedrock_path} ({len(dev_bedrock_records)} records)")

    # 4. Upload dev Bedrock input to S3 for the parallel Bedrock batch job
    if args.data_bucket:
        prefix = f"{args.model_name_prefix}/" if args.model_name_prefix else ""
        dev_bedrock_s3_key = f"{prefix}dev-labeling/input/dev_bedrock_input.jsonl"
        with open(dev_bedrock_path, 'rb') as f:
            dev_bedrock_s3_path = upload_to_s3(s3, args.data_bucket, dev_bedrock_s3_key, f.read())
        logger.info(f"Dev Bedrock input uploaded to {dev_bedrock_s3_path}")

        # Also upload dev_data.jsonl to S3 for use by training job
        dev_data_s3_key = f"{prefix}dev-labeling/dev_data.jsonl"
        with open(dev_path, 'rb') as f:
            dev_data_s3_path = upload_to_s3(s3, args.data_bucket, dev_data_s3_key, f.read())
        logger.info(f"Dev data uploaded to {dev_data_s3_path}")
    else:
        dev_bedrock_s3_path = ""
        dev_data_s3_path = ""

    # 5. Dev metadata
    dev_meta = {
        "num_dev_queries": len(dev_samples),
        "num_dev_bedrock_records": len(dev_bedrock_records),
        "dev_bm25_top_k": dev_top_k,
        "num_train_docs": len(train_doc_ids),
        "num_dev_docs": len(dev_doc_ids),
        "num_train_samples": len(training_samples),
        "dev_bedrock_s3_path": dev_bedrock_s3_path,
        "dev_data_s3_path": dev_data_s3_path,
    }
    dev_meta_path = output_dir / "dev_meta.json"
    with open(dev_meta_path, 'w', encoding='utf-8') as f:
        json.dump(dev_meta, f, indent=2)
    logger.info(f"Dev metadata written to {dev_meta_path}")


if __name__ == "__main__":
    main()
