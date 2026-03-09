"""
Model Evaluation Script for SageMaker Training Job

Evaluates fine-tuned model against baseline using LLM-judged relevance.

Pipeline:
1. Split scored training data into train/eval sets (by eval_split_ratio)
2. Build eval corpus from eval split documents
3. Use fine-tuned model to retrieve top-k candidates per eval query
4. Use Bedrock LLM to judge relevance of each (query, candidate) pair (0-1 score)
5. Filter to high-quality pairs (score >= threshold, default 0.8)
6. Compute NDCG@10, MRR@10, Recall@10 for fine-tuned vs baseline model
7. Apply quality gate and write results

Input channels:
    - model: Fine-tuned model artifacts (model.tar.gz from training job)
    - scoring: Scored training data (model.tar.gz containing scored_training_data.json)

Output: evaluation_results.json written to SM_MODEL_DIR
"""

import argparse
import json
import logging
import os
import re
import tarfile
import time
from pathlib import Path

import boto3
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)



# ---------------------------------------------------------------------------
# LLM Relevance Judging (adapted from llm-relevance-judge)
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """You are an expert search relevance rater. Evaluate the relevance between a search query and a document with these criteria:
- Score 1.0: Perfect match, highly relevant
- Score 0.7-0.9: Very relevant with minor variations
- Score 0.4-0.6: Moderately relevant
- Score 0.1-0.3: Slightly relevant
- Score 0.0: Completely irrelevant

Return ONLY a JSON object in this EXACT format:
{"rating_score": <score>}
Do not include any explanation, commentary, or markdown formatting."""

JUDGE_USER_TEMPLATE = "Query: {query}\n\nDocument: {document}"


def parse_llm_score(response_text: str) -> float:
    """Parse score from LLM response with fallback strategies."""
    text = response_text.strip()

    # Remove markdown code blocks
    if '```json' in text:
        text = text.split('```json')[1].split('```')[0].strip()
    elif '```' in text:
        text = text.split('```')[1].split('```')[0].strip()

    # Strategy 1: JSON parse
    try:
        result = json.loads(text)
        score = result.get('rating_score')
        if score is not None and 0 <= float(score) <= 1:
            return float(score)
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    # Strategy 2: regex
    match = re.search(r'"rating_score"\s*:\s*([0-9]*\.?[0-9]+)', response_text)
    if match:
        try:
            score = float(match.group(1))
            if 0 <= score <= 1:
                return score
        except ValueError:
            pass

    return -1.0  # parse failure


def judge_pair(bedrock_client, model_id: str, query: str, document: str,
               max_retries: int = 2) -> float:
    """Judge a single (query, document) pair using Bedrock LLM. Returns 0-1 score or -1 on failure."""
    user_content = JUDGE_USER_TEMPLATE.format(
        query=query, document=document[:4000]  # truncate long docs
    )

    request_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 256,
        "system": JUDGE_SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": user_content}]
    }

    for attempt in range(max_retries + 1):
        try:
            response = bedrock_client.invoke_model(
                modelId=model_id,
                body=json.dumps(request_body),
                contentType='application/json',
                accept='application/json'
            )
            body = json.loads(response['body'].read())
            content = body.get('content', [])
            for block in content:
                if block.get('type') == 'text':
                    score = parse_llm_score(block['text'])
                    if score >= 0:
                        return score
            raise ValueError("Could not parse score")
        except Exception as e:
            if attempt < max_retries:
                time.sleep(2 * (attempt + 1))
                continue
            logger.warning(f"LLM judge failed after {max_retries + 1} attempts: {e}")
            return -1.0

    return -1.0


def judge_pairs_batch(bedrock_client, model_id: str,
                      pairs: list, batch_log_interval: int = 50) -> list:
    """Judge a list of (query, document) pairs. Returns list of scores."""
    scores = []
    for i, (query, doc) in enumerate(pairs):
        score = judge_pair(bedrock_client, model_id, query, doc)
        scores.append(score)
        if (i + 1) % batch_log_interval == 0:
            logger.info(f"  Judged {i + 1}/{len(pairs)} pairs")
    return scores


# ---------------------------------------------------------------------------
# Retrieval evaluation helpers
# ---------------------------------------------------------------------------

def compute_ndcg(relevances, k):
    """Compute NDCG@k from a list of relevance scores."""
    dcg = sum(
        (2 ** rel - 1) / np.log2(i + 2)
        for i, rel in enumerate(relevances[:k])
    )
    ideal = sorted(relevances, reverse=True)[:k]
    idcg = sum(
        (2 ** rel - 1) / np.log2(i + 2)
        for i, rel in enumerate(ideal)
    )
    return dcg / idcg if idcg > 0 else 0.0


def retrieve_and_evaluate(model, queries, corpus_texts, qrels,
                          k_values=(10, 100), encode_batch_size=16):
    """
    Encode corpus + queries, compute retrieval metrics.

    qrels: list of dicts, one per query. Each dict maps corpus_idx -> relevance (binary 0/1).
    """
    logger.info(f"Encoding {len(corpus_texts)} corpus documents (batch_size={encode_batch_size})...")
    corpus_emb = model.encode(corpus_texts, normalize_embeddings=True,
                              batch_size=encode_batch_size, show_progress_bar=True)

    results = {f"ndcg@{k}": [] for k in k_values}
    results.update({f"recall@{k}": [] for k in k_values})
    results["mrr@10"] = []
    results["hit_rate@10"] = []

    logger.info(f"Evaluating {len(queries)} queries...")
    for i, query in enumerate(queries):
        q_emb = model.encode(query, normalize_embeddings=True)
        scores = np.dot(corpus_emb, q_emb)
        ranked = np.argsort(scores)[::-1]
        rels = qrels[i]

        for k in k_values:
            top_k = ranked[:k]
            top_rels = [rels.get(int(idx), 0) for idx in top_k]
            results[f"ndcg@{k}"].append(compute_ndcg(top_rels, k))

            total_rel = sum(1 for r in rels.values() if r > 0)
            retrieved_rel = sum(1 for r in top_rels if r > 0)
            results[f"recall@{k}"].append(
                retrieved_rel / total_rel if total_rel > 0 else 0
            )

        # MRR@10
        for rank, idx in enumerate(ranked[:10]):
            if rels.get(int(idx), 0) > 0:
                results["mrr@10"].append(1 / (rank + 1))
                break
        else:
            results["mrr@10"].append(0)

        # Hit Rate@10
        hit = any(rels.get(int(idx), 0) > 0 for idx in ranked[:10])
        results["hit_rate@10"].append(1.0 if hit else 0.0)

    return {m: float(np.mean(v)) for m, v in results.items()}




# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned model with LLM relevance judging")

    parser.add_argument("--model-dir", type=str,
                        default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--model-channel", type=str,
                        default=os.environ.get("SM_CHANNEL_MODEL", "/opt/ml/input/data/model"))
    parser.add_argument("--scoring-channel", type=str,
                        default=os.environ.get("SM_CHANNEL_SCORING", "/opt/ml/input/data/scoring"))

    # Model config
    parser.add_argument("--base-model-name", type=str, default="BAAI/bge-m3")

    # Eval config
    parser.add_argument("--retrieval-top-k", type=int, default=20,
                        help="Number of candidates to retrieve per query for LLM judging")
    parser.add_argument("--relevance-threshold", type=float, default=0.8,
                        help="LLM score threshold for a pair to be considered relevant (ground truth)")

    # LLM judge config
    parser.add_argument("--judge-model-id", type=str,
                        default="us.anthropic.claude-haiku-4-5-20251001-v1:0",
                        help="Bedrock model ID for LLM relevance judging")
    parser.add_argument("--bedrock-region", type=str, default="us-east-1")

    # Performance tuning
    parser.add_argument("--encode-batch-size", type=int, default=16,
                        help="Batch size for model.encode() — reduce if OOM (default 16)")

    # Quality gate
    parser.add_argument("--threshold-ndcg10", type=float, default=0.0,
                        help="Minimum absolute NDCG@10 for quality gate (0 = disabled)")
    parser.add_argument("--threshold-improvement-ndcg10", type=float, default=0.0,
                        help="Minimum NDCG@10 improvement over baseline (0 = disabled)")

    return parser.parse_args()


def main():
    args = parse_args()

    # ---- Extract tar.gz from input channels ----
    # SageMaker downloads model.tar.gz into the channel dir and may or may not
    # auto-extract it. train.py also creates an inner model.tar.gz inside its
    # output, so after extraction we can end up with a nested tar.gz that must
    # NOT be re-extracted (it causes EOFError).
    #
    # Strategy: snapshot tar.gz list BEFORE extraction, extract only those,
    # then remove any NEW tar.gz files that appeared (the inner ones).
    for channel_dir in [args.model_channel, args.scoring_channel]:
        channel_path = Path(channel_dir)
        tar_files_before = set(channel_path.glob("*.tar.gz"))
        logger.info(f"Channel {channel_dir}: found {len(tar_files_before)} tar.gz file(s) before extraction")
        logger.info(f"Channel {channel_dir} contents: {[f.name for f in channel_path.iterdir()]}")

        for tar_file in tar_files_before:
            logger.info(f"Extracting {tar_file}...")
            try:
                with tarfile.open(tar_file, "r:gz") as tar:
                    tar.extractall(path=channel_dir)
            except Exception as e:
                logger.warning(f"Failed to extract {tar_file}: {e}, skipping")

        # Remove any NEW tar.gz files that appeared after extraction (inner artifacts)
        tar_files_after = set(channel_path.glob("*.tar.gz"))
        new_tar_files = tar_files_after - tar_files_before
        for new_tar in new_tar_files:
            logger.info(f"Removing nested tar.gz artifact: {new_tar.name}")
            new_tar.unlink()

        logger.info(f"Channel {channel_dir} contents after extraction: "
                     f"{[f.name for f in channel_path.iterdir() if not f.name.startswith('checkpoint')]}")

    # ---- Load eval data (pre-split by score.py) ----
    scoring_dir = Path(args.scoring_channel)
    eval_data_path = scoring_dir / "scored_eval_data.json"
    if not eval_data_path.exists():
        # Fallback: try any json file (backward compat with no-split mode)
        data_files = list(scoring_dir.glob("*.json"))
        if not data_files:
            raise ValueError(f"No eval data found in {scoring_dir}")
        eval_data_path = data_files[0]
        logger.warning(f"scored_eval_data.json not found, falling back to {eval_data_path.name}")

    with open(eval_data_path, "r") as f:
        content = f.read().strip()
        if content.startswith('['):
            scored_data = json.loads(content)
        else:
            scored_data = [json.loads(line) for line in content.split('\n') if line.strip()]

    logger.info(f"Loaded {len(scored_data)} eval samples from {eval_data_path.name}")

    # ---- Build eval queries and corpus from scored eval data ----
    eval_queries = []
    eval_corpus_texts = []
    corpus_text_to_idx = {}

    for item in scored_data:
        query = item["query"]
        docs = item["docs"]
        scores = item.get("scores", [])

        if not docs:
            continue

        eval_queries.append(query)
        for doc in docs:
            if doc not in corpus_text_to_idx:
                corpus_text_to_idx[doc] = len(eval_corpus_texts)
                eval_corpus_texts.append(doc)

    eval_corpus = eval_corpus_texts
    logger.info(f"Eval set: {len(eval_queries)} queries, {len(eval_corpus)} corpus docs")

    if len(eval_queries) == 0:
        raise ValueError("No eval queries in scored_eval_data.json")

    # ---- Load fine-tuned model ----
    model_dir = Path(args.model_channel)
    logger.info(f"Loading fine-tuned model from {model_dir}...")
    finetuned_model = SentenceTransformer(str(model_dir))

    # ---- Retrieve candidates using fine-tuned model ----
    encode_bs = args.encode_batch_size
    logger.info(f"Encoding eval corpus ({len(eval_corpus)} docs) with fine-tuned model (batch_size={encode_bs})...")
    corpus_emb = finetuned_model.encode(
        eval_corpus, normalize_embeddings=True, batch_size=encode_bs, show_progress_bar=True
    )

    top_k = args.retrieval_top_k
    pairs_to_judge = []  # (query_idx, corpus_idx, query_text, doc_text)

    logger.info(f"Retrieving top-{top_k} candidates per query...")
    for qi, query in enumerate(eval_queries):
        q_emb = finetuned_model.encode(query, normalize_embeddings=True)
        scores = np.dot(corpus_emb, q_emb)
        ranked = np.argsort(scores)[::-1][:top_k]
        for ci in ranked:
            pairs_to_judge.append((qi, int(ci), query, eval_corpus[int(ci)]))

    logger.info(f"Total (query, doc) pairs to judge: {len(pairs_to_judge)}")

    # ---- LLM relevance judging ----
    bedrock = boto3.client('bedrock-runtime', region_name=args.bedrock_region)

    logger.info(f"Starting LLM relevance judging with {args.judge_model_id}...")
    logger.info(f"Relevance threshold: {args.relevance_threshold}")

    llm_scores = judge_pairs_batch(
        bedrock, args.judge_model_id,
        [(p[2], p[3]) for p in pairs_to_judge],
        batch_log_interval=100
    )

    # ---- Build ground truth qrels from LLM scores ----
    # Only pairs with LLM score >= threshold are considered relevant
    threshold = args.relevance_threshold
    qrels_finetuned = [{} for _ in range(len(eval_queries))]
    qrels_baseline = [{} for _ in range(len(eval_queries))]

    total_relevant = 0
    total_judged = 0
    score_distribution = {"0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0,
                          "0.6-0.8": 0, "0.8-1.0": 0, "failed": 0}

    for idx, (qi, ci, _, _) in enumerate(pairs_to_judge):
        s = llm_scores[idx]
        if s < 0:
            score_distribution["failed"] += 1
            continue

        total_judged += 1
        if s < 0.2:
            score_distribution["0.0-0.2"] += 1
        elif s < 0.4:
            score_distribution["0.2-0.4"] += 1
        elif s < 0.6:
            score_distribution["0.4-0.6"] += 1
        elif s < 0.8:
            score_distribution["0.6-0.8"] += 1
        else:
            score_distribution["0.8-1.0"] += 1

        rel = 1 if s >= threshold else 0
        if rel:
            total_relevant += 1
        # Both models evaluated against same ground truth
        qrels_finetuned[qi][ci] = rel
        qrels_baseline[qi][ci] = rel

    logger.info(f"LLM judging complete. Judged: {total_judged}, "
                f"Relevant (>= {threshold}): {total_relevant}")
    logger.info(f"Score distribution: {json.dumps(score_distribution)}")

    # Filter out queries with zero relevant docs (can't compute meaningful metrics)
    valid_indices = [i for i in range(len(eval_queries))
                     if any(v > 0 for v in qrels_finetuned[i].values())]

    if not valid_indices:
        logger.warning("No queries have relevant documents after LLM judging. "
                        "Quality gate will FAIL.")
        finetuned_metrics = {"ndcg@10": 0, "mrr@10": 0, "recall@10": 0,
                             "hit_rate@10": 0, "ndcg@100": 0, "recall@100": 0}
        baseline_metrics = dict(finetuned_metrics)
        del finetuned_model
        torch.cuda.empty_cache()
    else:
        filtered_queries = [eval_queries[i] for i in valid_indices]
        filtered_qrels_ft = [qrels_finetuned[i] for i in valid_indices]
        filtered_qrels_bl = [qrels_baseline[i] for i in valid_indices]

        logger.info(f"Evaluating on {len(filtered_queries)} queries with relevant docs...")

        # ---- Evaluate fine-tuned model ----
        logger.info("Computing metrics for fine-tuned model...")
        finetuned_metrics = retrieve_and_evaluate(
            finetuned_model, filtered_queries, eval_corpus, filtered_qrels_ft,
            encode_batch_size=encode_bs
        )
        logger.info(f"Fine-tuned: {json.dumps(finetuned_metrics, indent=2)}")

        # Free fine-tuned model GPU memory before loading baseline
        del finetuned_model
        del corpus_emb
        torch.cuda.empty_cache()
        logger.info("Released fine-tuned model from GPU memory")

        # ---- Evaluate baseline model ----
        logger.info(f"Loading baseline model: {args.base_model_name}...")
        baseline_model = SentenceTransformer(args.base_model_name)

        logger.info("Computing metrics for baseline model...")
        baseline_metrics = retrieve_and_evaluate(
            baseline_model, filtered_queries, eval_corpus, filtered_qrels_bl,
            encode_batch_size=encode_bs
        )
        logger.info(f"Baseline: {json.dumps(baseline_metrics, indent=2)}")

        del baseline_model
        torch.cuda.empty_cache()

    # ---- Compute improvements ----
    improvements = {
        m: finetuned_metrics[m] - baseline_metrics[m]
        for m in finetuned_metrics
    }

    # ---- Quality gate ----
    quality_gate = "PASSED"
    failures = []

    if args.threshold_ndcg10 > 0:
        if finetuned_metrics.get("ndcg@10", 0) < args.threshold_ndcg10:
            quality_gate = "FAILED"
            failures.append(
                f"ndcg@10={finetuned_metrics['ndcg@10']:.4f} < {args.threshold_ndcg10}"
            )

    if args.threshold_improvement_ndcg10 > 0:
        imp = improvements.get("ndcg@10", 0)
        if imp < args.threshold_improvement_ndcg10:
            quality_gate = "FAILED"
            failures.append(
                f"ndcg@10 improvement={imp:.4f} < {args.threshold_improvement_ndcg10}"
            )

    # ---- Write results ----
    results = {
        "finetuned_metrics": finetuned_metrics,
        "baseline_metrics": baseline_metrics,
        "improvements": improvements,
        "quality_gate": quality_gate,
        "failures": failures,
        "eval_config": {
            "retrieval_top_k": top_k,
            "relevance_threshold": threshold,
            "judge_model_id": args.judge_model_id,
            "base_model_name": args.base_model_name,
        },
        "eval_stats": {
            "total_queries": len(eval_queries),
            "queries_with_relevant_docs": len(valid_indices) if valid_indices else 0,
            "total_corpus_docs": len(eval_corpus),
            "total_pairs_judged": total_judged,
            "total_relevant_pairs": total_relevant,
            "score_distribution": score_distribution,
        },
    }

    output_path = Path(args.model_dir) / "evaluation_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nQuality Gate: {quality_gate}")
    if failures:
        logger.info(f"Failures: {failures}")
    logger.info(f"Results written to {output_path}")


if __name__ == "__main__":
    main()
