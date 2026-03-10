"""
Evaluation utilities for dev set scoring during training.

Computes NDCG@10 (threshold=0.8) and HQ@10 (threshold=0.99) using
pre-computed LLM relevance labels as ground truth.

The model encodes dev queries and candidate documents, ranks by cosine
similarity, then metrics are computed over the LLM-labeled scores.
"""

import json
import math
import logging
from pathlib import Path

import torch
import numpy as np

logger = logging.getLogger(__name__)

# Metric thresholds
NDCG_THRESHOLD = 0.8
HQ_THRESHOLD = 0.99
TOP_K = 10


def calculate_dcg(scores):
    """Calculate Discounted Cumulative Gain."""
    dcg = 0.0
    for i, score in enumerate(scores):
        numerator = (2 ** score) - 1
        denominator = math.log2(i + 2)
        dcg += numerator / denominator
    return dcg


def calculate_ndcg(scores, k=TOP_K):
    """Calculate NDCG@k with thresholded scores."""
    scores = scores[:k]
    if not scores:
        return 0.0
    actual_dcg = calculate_dcg(scores)
    ideal_scores = sorted(scores, reverse=True)
    ideal_dcg = calculate_dcg(ideal_scores)
    if ideal_dcg == 0.0:
        return 0.0
    return actual_dcg / ideal_dcg


def calculate_hq_score(scores, k=TOP_K, threshold=HQ_THRESHOLD):
    """Calculate average number of high-quality results in top-k."""
    scores = scores[:k]
    return sum(1 for s in scores if s >= threshold)


def load_dev_data(dev_data_path):
    """Load dev data with queries and candidates.

    Returns list of dicts with keys: query, positive_id, candidates
    Each candidate has: id, text
    """
    samples = []
    with open(dev_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def load_dev_labels(dev_labels_path):
    """Load LLM relevance labels for dev set.

    Returns dict mapping (query, doc_id) -> llm_score
    """
    labels = {}
    with open(dev_labels_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            key = (record["query"], record["candidate_id"])
            labels[key] = record["llm_score"]
    return labels


def encode_texts(model, texts, batch_size=64):
    """Encode texts using sentence transformer model."""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        with torch.no_grad():
            embeddings = model.encode(batch, convert_to_tensor=True, show_progress_bar=False)
        all_embeddings.append(embeddings)
    return torch.cat(all_embeddings, dim=0)


def evaluate_model(model, dev_samples, dev_labels, device='cuda'):
    """Evaluate model on dev set using LLM labels as ground truth.

    For each dev query:
    1. Encode query and all candidates
    2. Rank candidates by cosine similarity
    3. Look up LLM labels for ranked candidates
    4. Compute NDCG@10(0.8) and HQ@10(0.99)

    Returns dict with metrics.
    """
    model.eval()

    all_ndcg_scores = []
    all_hq_scores = []
    n_missing_labels = 0
    n_total_pairs = 0

    for sample in dev_samples:
        query = sample["query"]
        candidates = sample["candidates"]

        if not candidates:
            continue

        # Encode query and candidates
        query_emb = encode_texts(model, [query])
        candidate_texts = [c["text"] for c in candidates]
        candidate_embs = encode_texts(model, candidate_texts)

        # Compute cosine similarities
        query_emb_norm = torch.nn.functional.normalize(query_emb, p=2, dim=1)
        candidate_embs_norm = torch.nn.functional.normalize(candidate_embs, p=2, dim=1)
        similarities = torch.mm(query_emb_norm, candidate_embs_norm.t()).squeeze(0)

        # Rank by similarity (descending)
        ranked_indices = similarities.argsort(descending=True).cpu().numpy()

        # Get LLM scores in ranked order
        ranked_llm_scores = []
        for idx in ranked_indices[:TOP_K]:
            cand_id = candidates[idx]["id"]
            key = (query, cand_id)
            n_total_pairs += 1
            if key in dev_labels:
                llm_score = dev_labels[key]
            else:
                llm_score = 0.0
                n_missing_labels += 1
            ranked_llm_scores.append(llm_score)

        # Apply threshold for NDCG: scores below threshold become 0
        thresholded_scores = [s if s >= NDCG_THRESHOLD else 0.0 for s in ranked_llm_scores]

        ndcg = calculate_ndcg(thresholded_scores, k=TOP_K)
        hq = calculate_hq_score(ranked_llm_scores, k=TOP_K, threshold=HQ_THRESHOLD)

        all_ndcg_scores.append(ndcg)
        all_hq_scores.append(hq)

    if not all_ndcg_scores:
        return {"ndcg@10": 0.0, "hq@10": 0.0, "num_queries": 0}

    avg_ndcg = sum(all_ndcg_scores) / len(all_ndcg_scores)
    avg_hq = sum(all_hq_scores) / len(all_hq_scores)

    if n_missing_labels > 0:
        logger.warning(f"Missing LLM labels for {n_missing_labels}/{n_total_pairs} pairs")

    return {
        "ndcg@10": round(avg_ndcg, 4),
        "hq@10": round(avg_hq, 4),
        "num_queries": len(all_ndcg_scores),
    }
