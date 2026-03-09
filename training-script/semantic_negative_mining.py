"""
Semantic Hard Negative Mining using Sentence Transformers

Uses a sentence transformer model to encode documents and queries,
builds an HNSW index with FP16 quantization for fast ANN search,
and mines hard negatives based on semantic similarity.

Supports multi-GPU encoding via sentence-transformers multi-process pool
and multiprocess COW (copy-on-write) for parallel HNSW search.

Input (via hyperparameters):
    --output-s3-path: S3 path to Bedrock batch output directory
    --documents-s3-path: S3 path to corpus documents JSONL
    --model-id: Sentence transformer model ID for encoding
    --num-negatives: Number of hard negatives per query (default: 20)

Output: training_data.jsonl written to SM_MODEL_DIR (/opt/ml/model/)

Usage:
    python semantic_negative_mining.py --output-s3-path s3://... --documents-s3-path s3://... --model-id BAAI/bge-m3
"""

import argparse
import json
import os
import re
import logging
import time
from pathlib import Path
from urllib.parse import urlparse

import boto3
import numpy as np
import torch
import torch.multiprocessing as mp
import faiss
from sentence_transformers import SentenceTransformer

# Let faiss use all cores for parallel HNSW search via OpenMP
faiss.omp_set_num_threads(os.cpu_count() or 1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)



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

        text = generated_text.strip()
        if text.startswith('```'):
            text = re.sub(r'^```\w*\n?', '', text)
            text = re.sub(r'\n?```$', '', text)

        parsed = json.loads(text)
        queries = parsed.get('queries', []) if isinstance(parsed, dict) else parsed
        return queries

    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', generated_text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
                return parsed.get('queries', [])
            except json.JSONDecodeError:
                pass
        return []


def _encode_worker(rank, world_size, sentences_shard, model_id, max_seq_length, batch_size, output_dir):
    """Worker: load model on cuda:{rank} and encode its shard of sentences."""
    device = f"cuda:{rank}"
    worker_logger = logging.getLogger(f"encode-worker-{rank}")
    worker_logger.info(f"Worker {rank}/{world_size} starting on {device}, {len(sentences_shard)} sentences")

    model = SentenceTransformer(model_id, device=device, trust_remote_code=True)
    model.max_seq_length = max_seq_length

    embeddings = model.encode(
        sentences_shard,
        batch_size=batch_size,
        device=device,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    out_path = os.path.join(output_dir, f"embeddings_{rank}.npy")
    np.save(out_path, embeddings)
    worker_logger.info(f"Worker {rank}: done, shape={embeddings.shape}, saved to {out_path}")


def encode_sentences(model_id, max_seq_length, sentences, batch_size, num_gpus):
    """
    Encode sentences with one process per GPU (spawn), each loading its own model.
    Returns L2-normalized float32 embeddings for HNSW inner-product search.
    """
    import tempfile

    if num_gpus > 1:
        logger.info(f"Multi-GPU encoding with {num_gpus} GPUs (spawn)")
        shard_size = (len(sentences) + num_gpus - 1) // num_gpus
        shards = [sentences[i * shard_size: (i + 1) * shard_size] for i in range(num_gpus)]

        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = mp.get_context('spawn')
            processes = []
            for rank in range(num_gpus):
                p = ctx.Process(
                    target=_encode_worker,
                    args=(rank, num_gpus, shards[rank], model_id, max_seq_length, batch_size, tmpdir),
                )
                p.start()
                processes.append(p)

            for p in processes:
                p.join()

            embeddings = np.concatenate(
                [np.load(os.path.join(tmpdir, f"embeddings_{rank}.npy")) for rank in range(num_gpus)],
                axis=0,
            )
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = SentenceTransformer(model_id, device=device, trust_remote_code=True)
        model.max_seq_length = max_seq_length
        embeddings = model.encode(
            sentences,
            batch_size=batch_size,
            device=device,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        del model

    embeddings = embeddings.astype(np.float32)
    faiss.normalize_L2(embeddings)
    return embeddings


def build_hnsw_index(embeddings, m=32, ef_construction=200):
    """
    Build HNSW index with FP16 scalar quantization and inner-product metric.

    Vectors are stored quantized to FP16 inside the index, cutting memory ~50%
    while maintaining search quality. Inner product on L2-normalized vectors
    is equivalent to cosine similarity.
    """
    dim = embeddings.shape[1]
    index = faiss.IndexHNSWSQ(
        dim, faiss.ScalarQuantizer.QT_fp16, m, faiss.METRIC_INNER_PRODUCT
    )
    index.hnsw.efConstruction = ef_construction

    logger.info(f"Training index (dim={dim}, M={m}, efConstruction={ef_construction}, metric=IP, quant=FP16)...")
    index.train(embeddings)

    logger.info(f"Adding {len(embeddings)} vectors to index...")
    index.add(embeddings)

    return index


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Mine semantic hard negatives using sentence transformers and HNSW"
    )

    # SageMaker environment variables
    parser.add_argument("--model-dir", type=str,
                        default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))

    # S3 paths
    parser.add_argument("--output-s3-path", type=str, required=True,
                        help="S3 path to Bedrock batch output directory")
    parser.add_argument("--documents-s3-path", type=str, required=True,
                        help="S3 path to corpus documents JSONL")

    # Model configuration
    parser.add_argument("--model-id", type=str, required=True,
                        help="Sentence transformer model ID for encoding")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size for encoding")
    parser.add_argument("--max-seq-length", type=int, default=512,
                        help="Maximum sequence length for encoding")

    # Mining parameters
    parser.add_argument("--num-negatives", type=int, default=20,
                        help="Number of hard negatives per query")
    parser.add_argument("--max-corpus-documents", type=int, default=0,
                        help="Max documents for corpus (0 = unlimited)")

    # HNSW parameters
    parser.add_argument("--hnsw-m", type=int, default=32,
                        help="HNSW M parameter (number of neighbors per node)")
    parser.add_argument("--ef-construction", type=int, default=200,
                        help="HNSW efConstruction parameter")
    parser.add_argument("--ef-search", type=int, default=128,
                        help="HNSW efSearch parameter")

    return parser.parse_args()


def main():
    args = parse_args()
    s3 = boto3.client('s3')

    # ── Load corpus documents ──────────────────────────────────────────
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

    if args.max_corpus_documents > 0 and len(doc_texts) > args.max_corpus_documents:
        logger.info(f"Truncating corpus from {len(doc_texts)} to {args.max_corpus_documents} documents")
        doc_ids = doc_ids[:args.max_corpus_documents]
        doc_texts = doc_texts[:args.max_corpus_documents]
        doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}

    del documents
    logger.info(f"Loaded {len(doc_texts)} documents in {time.time() - t0:.1f}s")

    # ── Parse Bedrock output ───────────────────────────────────────────
    output_bucket, output_prefix = parse_s3_path(args.output_s3_path)
    paginator = s3.get_paginator('list_objects_v2')
    output_files = []
    for page in paginator.paginate(Bucket=output_bucket, Prefix=output_prefix):
        for obj in page.get('Contents', []):
            if obj['Key'].endswith('.jsonl.out'):
                output_files.append(obj['Key'])

    logger.info(f"Found {len(output_files)} Bedrock output files")

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

    all_queries = [p["query"] for p in query_doc_pairs]
    positive_doc_ids = [p["doc_id"] for p in query_doc_pairs]
    positive_doc_texts = [p["doc_text"] for p in query_doc_pairs]

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    logger.info(f"Detected {num_gpus} GPU(s)")

    # ── Encode corpus documents (multi-GPU spawn) ──────────────────────
    t0 = time.time()
    logger.info(f"Encoding {len(doc_texts)} corpus documents with model {args.model_id}...")
    doc_embeddings = encode_sentences(
        args.model_id, args.max_seq_length, doc_texts, args.batch_size, num_gpus
    )
    logger.info(f"Corpus encoding completed in {time.time() - t0:.1f}s, shape: {doc_embeddings.shape}")

    # ── Encode queries (multi-GPU spawn) ───────────────────────────────
    t0 = time.time()
    logger.info(f"Encoding {len(all_queries)} queries...")
    query_embeddings = encode_sentences(
        args.model_id, args.max_seq_length, all_queries, args.batch_size, num_gpus
    )
    logger.info(f"Query encoding completed in {time.time() - t0:.1f}s, shape: {query_embeddings.shape}")

    # ── Build HNSW index with FP16 quantization ───────────────────────
    t0 = time.time()
    logger.info("Building HNSW index with FP16 quantization...")
    index = build_hnsw_index(
        doc_embeddings, m=args.hnsw_m, ef_construction=args.ef_construction
    )
    del doc_embeddings
    logger.info(f"HNSW index built in {time.time() - t0:.1f}s")

    # ── Search (faiss uses OpenMP threads internally for parallelism) ──
    num_negatives = args.num_negatives
    top_k = min(num_negatives + 5, len(doc_texts))
    index.hnsw.efSearch = args.ef_search

    n_queries = len(all_queries)
    logger.info(f"Searching top-{top_k} for {n_queries} queries (efSearch={args.ef_search}, omp_threads={faiss.omp_get_max_threads()})...")
    t0 = time.time()
    _scores, all_indices = index.search(query_embeddings, top_k)
    del _scores
    logger.info(f"Search completed in {time.time() - t0:.1f}s")

    # ── Build training samples ─────────────────────────────────────────
    training_samples = []
    for i, (query, pos_doc_id, pos_doc_text) in enumerate(
        zip(all_queries, positive_doc_ids, positive_doc_texts)
    ):
        pos_idx = doc_id_to_idx.get(pos_doc_id, -1)
        hard_negatives = []
        for j in range(all_indices.shape[1]):
            doc_idx = int(all_indices[i, j])
            if doc_idx != pos_idx and doc_idx >= 0:
                hard_negatives.append(doc_texts[doc_idx])
                if len(hard_negatives) >= num_negatives:
                    break
        training_samples.append({
            "anchor": query,
            "positive": pos_doc_text,
            "negatives": hard_negatives
        })

        if (i + 1) % 10000 == 0:
            logger.info(f"Processed {i + 1}/{n_queries} queries")

    logger.info(f"Created {len(training_samples)} training samples with hard negatives")

    # ── Write output ───────────────────────────────────────────────────
    output_dir = Path(args.model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "training_data.jsonl"

    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in training_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    logger.info(f"Training data written to {output_path} ({len(training_samples)} samples)")


if __name__ == "__main__":
    main()
