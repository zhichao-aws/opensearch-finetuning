"""
Evaluation Checker Lambda

Reads evaluation_results.json from SageMaker job output (model.tar.gz)
and returns the quality gate decision for Step Functions.

Input:
{
    "evaluation_output_s3_path": "s3://bucket/model-name/eval-output/.../model.tar.gz"
}

Output:
{
    "quality_gate": "PASSED" | "FAILED",
    "metrics": { ... },
    "failures": [...],
    "eval_stats": { ... }
}
"""

import json
import os
import tarfile

import boto3
from urllib.parse import urlparse


def handler(event, context):
    s3_path = event['evaluation_output_s3_path']
    parsed = urlparse(s3_path)
    bucket = parsed.netloc
    key = parsed.path.lstrip('/')

    s3 = boto3.client('s3')
    tmp_tar = '/tmp/model.tar.gz'
    tmp_dir = '/tmp/eval'

    # Download and extract
    s3.download_file(bucket, key, tmp_tar)

    os.makedirs(tmp_dir, exist_ok=True)
    with tarfile.open(tmp_tar, 'r:gz') as tar:
        tar.extractall(path=tmp_dir)

    # Read evaluation results
    results_path = os.path.join(tmp_dir, 'evaluation_results.json')
    with open(results_path, 'r') as f:
        results = json.load(f)

    return {
        "quality_gate": results["quality_gate"],
        "metrics": {
            "finetuned": results["finetuned_metrics"],
            "baseline": results.get("baseline_metrics", {}),
            "improvements": results.get("improvements", {}),
        },
        "failures": results.get("failures", []),
        "eval_stats": results.get("eval_stats", {}),
    }
