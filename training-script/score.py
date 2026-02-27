"""
Teacher Score Generation using Cross-Encoder

Uses BGE-reranker-v2.5-gemma2-lightweight to generate teacher scores for
knowledge distillation training.

Input: training_data.jsonl with format:
    {"anchor": "query", "positive": "doc_text", "negatives": ["neg1", "neg2", ...]}

Output: scored_training_data.json with format:
    [{"query": "...", "docs": ["pos", "neg1", ...], "scores": [s1, s2, ...]}, ...]

Usage:
    python score.py --training-dir /path/to/input --output-data-dir /path/to/output
"""

import json
import os
import argparse
import logging
import tarfile
from pathlib import Path

import torch
import torch.multiprocessing as mp
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def last_logit_pool(logits: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Extract the last token logit for scoring."""
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return logits[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = logits.shape[0]
        return torch.stack([logits[i, sequence_lengths[i]] for i in range(batch_size)], dim=0)


def get_inputs(pairs, tokenizer, prompt=None, max_length=1024):
    """Prepare inputs for the reranker model."""
    if prompt is None:
        prompt = "Predict whether passage B contains an answer to query A."
    sep = "\n"
    prompt_inputs = tokenizer(prompt,
                              return_tensors=None,
                              add_special_tokens=False)['input_ids']
    sep_inputs = tokenizer(sep,
                           return_tensors=None,
                           add_special_tokens=False)['input_ids']
    inputs = []
    query_lengths = []
    prompt_lengths = []

    for query, passage in pairs:
        query_inputs = tokenizer(f'A: {query}',
                                 return_tensors=None,
                                 add_special_tokens=False,
                                 max_length=max_length * 3 // 4,
                                 truncation=True)
        passage_inputs = tokenizer(f'B: {passage}',
                                   return_tensors=None,
                                   add_special_tokens=False,
                                   max_length=max_length,
                                   truncation=True)
        item = tokenizer.prepare_for_model(
            [tokenizer.bos_token_id] + query_inputs['input_ids'],
            sep_inputs + passage_inputs['input_ids'],
            truncation='only_second',
            max_length=max_length,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            add_special_tokens=False
        )
        item['input_ids'] = item['input_ids'] + sep_inputs + prompt_inputs
        item['attention_mask'] = [1] * len(item['input_ids'])
        inputs.append(item)
        query_lengths.append(len([tokenizer.bos_token_id] + query_inputs['input_ids'] + sep_inputs))
        prompt_lengths.append(len(sep_inputs + prompt_inputs))

    return tokenizer.pad(
        inputs,
        padding=True,
        max_length=max_length + len(sep_inputs) + len(prompt_inputs),
        pad_to_multiple_of=8,
        return_tensors='pt',
    ), query_lengths, prompt_lengths


def load_training_data(data_dir: str, max_negatives: int = 20):
    """
    Load training data from JSONL files.

    Input format: {"anchor": "query", "positive": "doc", "negatives": ["neg1", ...]}
    Returns list of (query, docs_list) tuples where docs_list = [positive] + negatives
    """
    data_path = Path(data_dir)
    jsonl_files = list(data_path.glob("*.jsonl")) + list(data_path.glob("*.json"))

    if not jsonl_files:
        raise ValueError(f"No JSONL files found in {data_dir}")

    logger.info(f"Found {len(jsonl_files)} data files: {[f.name for f in jsonl_files]}")

    samples = []
    for jsonl_file in jsonl_files:
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                query = data["anchor"]
                positive = data["positive"]
                negatives = data.get("negatives", [])[:max_negatives]
                docs = [positive] + negatives
                samples.append((query, docs))

    logger.info(f"Loaded {len(samples)} samples")
    return samples


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate teacher scores using cross-encoder")

    # SageMaker environment variables
    parser.add_argument("--training-dir", type=str,
                        default=os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training"))
    parser.add_argument("--output-data-dir", type=str,
                        default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--model-dir", type=str,
                        default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))

    # Model configuration
    parser.add_argument("--reranker-model", type=str,
                        default="BAAI/bge-reranker-v2.5-gemma2-lightweight",
                        help="Cross-encoder model for scoring")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for scoring")
    parser.add_argument("--max-length", type=int, default=1024, help="Max sequence length")
    parser.add_argument("--max-negatives", type=int, default=4, help="Max negatives per query")

    return parser.parse_args()


def score_worker(rank: int, world_size: int, samples: list, args, return_dict: dict):
    """
    Worker function for multi-GPU scoring.
    Each worker loads model on its assigned GPU and processes its data shard.
    """
    device = torch.device(f"cuda:{rank}")

    # Setup logging for this worker
    worker_logger = logging.getLogger(f"worker-{rank}")
    worker_logger.info(f"Worker {rank}/{world_size} starting on {device}")

    # Load tokenizer and model on this GPU
    tokenizer = AutoTokenizer.from_pretrained(args.reranker_model, trust_remote_code=True)
    tokenizer.padding_side = 'right'

    model = AutoModelForCausalLM.from_pretrained(
        args.reranker_model,
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    model = model.to(device)
    model.eval()
    worker_logger.info(f"Worker {rank}: Model loaded on {device}")

    # Shard samples across workers
    shard_samples = samples[rank::world_size]
    worker_logger.info(f"Worker {rank}: Processing {len(shard_samples)} samples")

    # Flatten to pairs for batch scoring
    all_pairs = []
    pair_to_local_idx = []
    for local_idx, (query, docs) in enumerate(shard_samples):
        for doc in docs:
            all_pairs.append([query, doc])
            pair_to_local_idx.append(local_idx)

    # Score in batches
    all_scores = []
    batch_size = args.batch_size

    with torch.no_grad():
        for i in range(0, len(all_pairs), batch_size):
            batch = all_pairs[i:i + batch_size]
            inputs, query_lengths, prompt_lengths = get_inputs(batch, tokenizer, max_length=args.max_length)
            inputs = inputs.to(device)

            outputs = model(
                **inputs,
                return_dict=True,
                cutoff_layers=[28],
                compress_ratio=2,
                compress_layer=[24, 40],
                query_lengths=query_lengths,
                prompt_lengths=prompt_lengths
            )

            for j in range(len(outputs.logits)):
                logits = last_logit_pool(outputs.logits[j], outputs.attention_masks[j])
                all_scores.extend(logits.cpu().float().tolist())

            del outputs
            del inputs

            if (i // batch_size + 1) % 100 == 0:
                worker_logger.info(f"Worker {rank}: Scored {min(i + batch_size, len(all_pairs))}/{len(all_pairs)} pairs")

    # Reconstruct samples with scores for this shard
    shard_results = []
    score_idx = 0
    for local_idx, (query, docs) in enumerate(shard_samples):
        sample_scores = all_scores[score_idx:score_idx + len(docs)]
        score_idx += len(docs)

        # Store with original global index for proper ordering
        global_idx = rank + local_idx * world_size
        shard_results.append({
            "global_idx": global_idx,
            "query": query,
            "docs": docs,
            "scores": sample_scores
        })

    worker_logger.info(f"Worker {rank}: Completed scoring {len(shard_results)} samples")
    return_dict[rank] = shard_results


def main():
    args = parse_args()

    # Detect number of GPUs
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    logger.info(f"Detected {num_gpus} GPU(s)")

    # Extract tar.gz files if present (SageMaker model artifacts from prior job)
    training_dir = Path(args.training_dir)
    tar_files = list(training_dir.glob("*.tar.gz"))
    for tar_file in tar_files:
        logger.info(f"Extracting {tar_file}...")
        with tarfile.open(tar_file, "r:gz") as tar:
            tar.extractall(path=training_dir)
        logger.info(f"Extracted {tar_file} to {training_dir}")

    # Load training data
    samples = load_training_data(args.training_dir, args.max_negatives)

    if num_gpus <= 1:
        # Single GPU or CPU mode
        logger.info("Running in single GPU/CPU mode")
        results = run_single_gpu(samples, args)
    else:
        # Multi-GPU mode
        logger.info(f"Running in multi-GPU mode with {num_gpus} GPUs")
        results = run_multi_gpu(samples, args, num_gpus)

    # Write output
    output_dir = Path(args.output_data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "scored_training_data.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(f"Results saved to: {output_path}")
    logger.info(f"Total samples: {len(results)}")


def run_single_gpu(samples: list, args) -> list:
    """Run scoring on single GPU or CPU."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logger.info(f"Loading reranker model: {args.reranker_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.reranker_model, trust_remote_code=True)
    tokenizer.padding_side = 'right'

    model = AutoModelForCausalLM.from_pretrained(
        args.reranker_model,
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    model = model.to(device)
    model.eval()
    logger.info(f"Model loaded successfully on {device}")

    # Flatten to pairs for batch scoring
    all_pairs = []
    for query, docs in samples:
        for doc in docs:
            all_pairs.append([query, doc])

    logger.info(f"Total pairs to score: {len(all_pairs)}")

    # Score in batches
    all_scores = []
    batch_size = args.batch_size

    with torch.no_grad():
        for i in range(0, len(all_pairs), batch_size):
            batch = all_pairs[i:i + batch_size]
            inputs, query_lengths, prompt_lengths = get_inputs(batch, tokenizer, max_length=args.max_length)
            inputs = inputs.to(device)

            outputs = model(
                **inputs,
                return_dict=True,
                cutoff_layers=[28],
                compress_ratio=2,
                compress_layer=[24, 40],
                query_lengths=query_lengths,
                prompt_lengths=prompt_lengths
            )

            for j in range(len(outputs.logits)):
                logits = last_logit_pool(outputs.logits[j], outputs.attention_masks[j])
                all_scores.extend(logits.cpu().float().tolist())

            del outputs
            del inputs

            if (i // batch_size + 1) % 100 == 0:
                logger.info(f"Scored {min(i + batch_size, len(all_pairs))}/{len(all_pairs)} pairs")

    logger.info(f"Scoring complete. Total scores: {len(all_scores)}")

    # Reconstruct samples with scores
    results = []
    score_idx = 0
    for query, docs in samples:
        sample_scores = all_scores[score_idx:score_idx + len(docs)]
        score_idx += len(docs)

        results.append({
            "query": query,
            "docs": docs,
            "scores": sample_scores
        })

    return results


def run_multi_gpu(samples: list, args, num_gpus: int) -> list:
    """Run scoring distributed across multiple GPUs."""
    # Use spawn method for CUDA compatibility
    mp.set_start_method('spawn', force=True)

    # Create a manager for sharing results between processes
    manager = mp.Manager()
    return_dict = manager.dict()

    # Spawn worker processes
    processes = []
    for rank in range(num_gpus):
        p = mp.Process(
            target=score_worker,
            args=(rank, num_gpus, samples, args, return_dict)
        )
        p.start()
        processes.append(p)

    # Wait for all processes to complete
    for p in processes:
        p.join()

    # Gather results from all workers
    all_results = []
    for rank in range(num_gpus):
        if rank in return_dict:
            all_results.extend(return_dict[rank])

    # Sort by global index to maintain original order
    all_results.sort(key=lambda x: x["global_idx"])

    # Remove global_idx from final output
    results = [
        {"query": r["query"], "docs": r["docs"], "scores": r["scores"]}
        for r in all_results
    ]

    logger.info(f"Gathered {len(results)} results from {num_gpus} GPUs")
    return results


if __name__ == "__main__":
    main()
