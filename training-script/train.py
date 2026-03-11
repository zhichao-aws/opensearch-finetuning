"""
KL Divergence Distillation Training with Dev Set Evaluation

Trains a sentence transformer model using knowledge distillation from
teacher scores (cross-encoder). Evaluates on dev set every N steps
using pre-computed LLM labels. Supports early stopping.

Input:
    - training channel: scored_training_data.json
    - dev channel: dev_data.jsonl + dev_labels.jsonl (LLM relevance scores)

Output: Fine-tuned sentence transformer model (best checkpoint by dev NDCG@10)

Usage:
    python train.py --training-dir /path/to/input --dev-dir /path/to/dev --model-dir /path/to/output
"""

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import torch
from accelerate import Accelerator
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
    util,
)
from transformers import TrainerCallback

from evaluate_utils import evaluate_model, load_dev_data, load_dev_labels

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)


class DevEvalCallback(TrainerCallback):
    """Callback to evaluate on dev set during training and support early stopping."""

    def __init__(self, model, dev_samples, dev_labels, eval_steps=100,
                 early_stopping_patience=3, model_dir="/opt/ml/model"):
        self.model = model
        self.dev_samples = dev_samples
        self.dev_labels = dev_labels
        self.eval_steps = eval_steps
        self.early_stopping_patience = early_stopping_patience
        self.model_dir = model_dir

        self.best_ndcg = -1.0
        self.best_step = 0
        self.no_improve_count = 0
        self.eval_history = []
        self.baseline_metrics = None
        self.should_stop = False

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.eval_steps != 0 or state.global_step == 0:
            return

        metrics = evaluate_model(self.model, self.dev_samples, self.dev_labels)
        metrics["step"] = state.global_step

        if self.baseline_metrics is None:
            # First eval is treated as baseline (model has barely trained)
            pass

        self.eval_history.append(metrics)

        ndcg = metrics["ndcg@10"]
        hq = metrics["hq@10"]
        print(f"\n[Dev Eval] Step {state.global_step}: NDCG@10(0.8)={ndcg:.4f}, HQ@10(0.99)={hq:.4f}")

        if ndcg > self.best_ndcg:
            self.best_ndcg = ndcg
            self.best_step = state.global_step
            self.no_improve_count = 0
            # Save best model
            best_dir = os.path.join(self.model_dir, "best_model")
            os.makedirs(best_dir, exist_ok=True)
            self.model.save_pretrained(best_dir)
            print(f"[Dev Eval] New best! Saved to {best_dir}")
        else:
            self.no_improve_count += 1
            print(f"[Dev Eval] No improvement ({self.no_improve_count}/{self.early_stopping_patience})")

        if self.no_improve_count >= self.early_stopping_patience:
            print(f"[Dev Eval] Early stopping at step {state.global_step} (best was step {self.best_step})")
            self.should_stop = True
            control.should_training_stop = True

    def on_train_begin(self, args, state, control, **kwargs):
        """Evaluate baseline (step 0) before any training."""
        print("\n[Dev Eval] Evaluating baseline model (step 0)...")
        metrics = evaluate_model(self.model, self.dev_samples, self.dev_labels)
        metrics["step"] = 0
        self.baseline_metrics = metrics
        self.eval_history.append(metrics)
        print(f"[Dev Eval] Baseline: NDCG@10(0.8)={metrics['ndcg@10']:.4f}, HQ@10(0.99)={metrics['hq@10']:.4f}")

    def get_report(self):
        """Generate evaluation report."""
        report = {
            "baseline": self.baseline_metrics,
            "best_step": self.best_step,
            "best_ndcg@10": self.best_ndcg,
            "early_stopped": self.should_stop,
            "history": self.eval_history,
        }
        if self.baseline_metrics:
            baseline_ndcg = self.baseline_metrics["ndcg@10"]
            if baseline_ndcg > 0:
                report["ndcg_improvement"] = f"{((self.best_ndcg - baseline_ndcg) / baseline_ndcg) * 100:.1f}%"
            else:
                report["ndcg_improvement"] = "N/A (baseline=0)"
        return report


def load_and_convert(data_path: str, num_negatives: int, teacher_score_scale_factor: float, topN: int = None):
    """
    Load scored training data and convert to format for DistillKLDivLoss.

    Input format: {"query": "...", "docs": ["pos", "neg1", ...], "scores": [s1, s2, ...]}
    Output: Dataset with columns: query, positive, negative1..N, label (tensor of scores)
    """
    queries, positives, labels = [], [], []
    negatives_cols = {f"negative{i+1}": [] for i in range(num_negatives)}

    n_skipped = 0

    # Handle both JSON array and JSONL format
    with open(data_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if content.startswith('['):
            data_array = json.loads(content)
        else:
            data_array = [json.loads(line) for line in content.split('\n') if line.strip()]

    for ex in data_array:
        q = ex["query"]
        docs = ex["docs"]
        scores = ex["scores"]

        if not docs or not scores or len(docs) != len(scores):
            n_skipped += 1
            continue

        # Need at least 1 positive + K negatives
        if len(docs) < 1 + num_negatives:
            n_skipped += 1
            continue

        # Select document with highest teacher score as positive
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        pos_doc = docs[best_idx]
        pos_score = float(scores[best_idx])

        # Sort remaining docs by score (hardest negatives)
        rest = [(docs[i], float(scores[i])) for i in range(len(docs)) if i != best_idx]
        rest.sort(key=lambda x: x[1], reverse=True)

        # Apply topN filter if specified
        if topN is not None:
            rest = rest[:topN]

        # Create training samples in chunks of num_negatives
        offset = 0
        while offset + num_negatives <= len(rest):
            chunk = rest[offset:offset + num_negatives]
            offset += num_negatives

            queries.append(q)
            positives.append(pos_doc)
            # Label: [score_pos, score_neg1, ..., score_negK]
            labels.append(
                [pos_score * teacher_score_scale_factor]
                + [s * teacher_score_scale_factor for (_, s) in chunk]
            )
            for i, (neg_doc, _) in enumerate(chunk):
                negatives_cols[f"negative{i+1}"].append(neg_doc)

    data = {"query": queries, "positive": positives, **negatives_cols, "label": labels}
    ds = Dataset.from_dict(data)

    # Convert labels to tensors
    ds = ds.map(lambda b: {"label": torch.tensor(b["label"], dtype=torch.float32)}, batched=False)
    print(f"Loaded {len(ds)} training examples; skipped {n_skipped}")
    return ds


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train sentence transformer with KL divergence distillation")

    # SageMaker environment variables
    parser.add_argument("--training-dir", type=str,
                        default=os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training"))
    parser.add_argument("--dev-dir", type=str,
                        default=os.environ.get("SM_CHANNEL_DEV", "/opt/ml/input/data/dev"))
    parser.add_argument("--model-dir", type=str,
                        default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--output-data-dir", type=str,
                        default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"))

    # Model configuration
    parser.add_argument("--model-name", type=str, default="BAAI/bge-m3",
                        help="Base model to fine-tune")

    # Training hyperparameters
    parser.add_argument("--num-negatives", type=int, default=2,
                        help="Number of negatives per training sample")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature for KL divergence loss")
    parser.add_argument("--use-dot", action="store_true",
                        help="Use dot product instead of cosine similarity")
    parser.add_argument("--train-batch-size", type=int, default=4,
                        help="Per-device training batch size")
    parser.add_argument("--total-batch-size", type=int, default=128,
                        help="Target total batch size (used to auto-calculate gradient accumulation)")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=-1,
                        help="Gradient accumulation steps (-1 to auto-calculate from total-batch-size)")
    parser.add_argument("--teacher-score-scale-factor", type=float, default=0.025,
                        help="Scaling factor for teacher scores")
    parser.add_argument("--learning-rate", type=float, default=5e-6,
                        help="Learning rate")
    parser.add_argument("--num-epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--max-steps", type=int, default=-1,
                        help="Max training steps (-1 for unlimited, trains full epochs)")
    parser.add_argument("--max-seq-length", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--warmup-ratio", type=float, default=0.1,
                        help="Warmup ratio")
    parser.add_argument("--logging-steps", type=int, default=50,
                        help="Logging frequency")
    parser.add_argument("--save-steps", type=int, default=500,
                        help="Checkpoint save frequency")
    parser.add_argument("--save-total-limit", type=int, default=2,
                        help="Max checkpoints to keep")
    parser.add_argument("--topN", type=int, default=None,
                        help="Only keep top N negatives by teacher score")

    # Dev evaluation
    parser.add_argument("--eval-steps", type=int, default=100,
                        help="Evaluate on dev set every N steps")
    parser.add_argument("--early-stopping-patience", type=int, default=3,
                        help="Stop after N evals without NDCG improvement")

    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize Accelerator for distributed training support
    accelerator = Accelerator()

    # Find training data file
    training_dir = Path(args.training_dir)

    # Check for tar.gz files and extract them (SageMaker passes model artifacts as tar.gz)
    # Only do this on main process to avoid race conditions
    if accelerator.is_main_process:
        for src_dir in [training_dir, Path(args.dev_dir)]:
            if src_dir.exists():
                tar_files = list(src_dir.glob("*.tar.gz"))
                for tar_file in tar_files:
                    print(f"Extracting {tar_file}...")
                    with tarfile.open(tar_file, "r:gz") as tar:
                        tar.extractall(path=src_dir)
                    print(f"Extracted {tar_file} to {src_dir}")

    # Wait for main process to finish extraction
    accelerator.wait_for_everyone()

    # Prefer scored_training_data.json specifically
    scored_file = training_dir / "scored_training_data.json"
    if scored_file.exists():
        data_path = scored_file
    else:
        data_files = list(training_dir.glob("*.json")) + list(training_dir.glob("*.jsonl"))
        if not data_files:
            raise ValueError(f"No training data found in {training_dir}")
        data_path = data_files[0]
    if accelerator.is_main_process:
        print(f"Using training data: {data_path}")

    # Load and convert data
    train_dataset = load_and_convert(
        str(data_path),
        args.num_negatives,
        args.teacher_score_scale_factor,
        args.topN
    )

    if len(train_dataset) == 0:
        raise ValueError("No valid training samples after processing")

    # Load dev data if available
    dev_dir = Path(args.dev_dir)
    dev_samples = None
    dev_labels = None

    if dev_dir.exists():
        dev_data_files = list(dev_dir.glob("dev_data.jsonl"))
        dev_label_files = list(dev_dir.glob("dev_labels.jsonl"))

        if dev_data_files and dev_label_files:
            if accelerator.is_main_process:
                print(f"Loading dev data from {dev_data_files[0]}")
                print(f"Loading dev labels from {dev_label_files[0]}")
            dev_samples = load_dev_data(str(dev_data_files[0]))
            dev_labels = load_dev_labels(str(dev_label_files[0]))
            if accelerator.is_main_process:
                print(f"Dev set: {len(dev_samples)} queries, {len(dev_labels)} labels")
        else:
            if accelerator.is_main_process:
                print(f"Dev data not found in {dev_dir}, skipping dev evaluation")

    # Use accelerator to get world size for gradient accumulation calculation
    num_processes = accelerator.num_processes
    if accelerator.is_main_process:
        print(f"Distributed training with {num_processes} process(es)")

    if args.gradient_accumulation_steps < 0:
        # Auto-calculate to achieve total_batch_size
        args.gradient_accumulation_steps = max(1, args.total_batch_size // (args.train_batch_size * num_processes))
        if accelerator.is_main_process:
            print(f"Auto-calculated gradient_accumulation_steps: {args.gradient_accumulation_steps}")

    effective_batch_size = args.train_batch_size * num_processes * args.gradient_accumulation_steps
    if accelerator.is_main_process:
        print(f"Effective batch size: {effective_batch_size} "
              f"(per_device={args.train_batch_size} x processes={num_processes} x accum={args.gradient_accumulation_steps})")

    # Load model
    if accelerator.is_main_process:
        print(f"Loading model: {args.model_name}")
    model = SentenceTransformer(args.model_name, trust_remote_code=True)
    model.max_seq_length = args.max_seq_length

    # Setup loss function
    similarity_fct = util.pairwise_cos_sim if not args.use_dot else util.pairwise_dot_score
    train_loss = losses.DistillKLDivLoss(
        model=model,
        similarity_fct=similarity_fct,
        temperature=args.temperature,
    )

    # Setup dev eval callback
    callbacks = []
    dev_callback = None
    if dev_samples and dev_labels:
        dev_callback = DevEvalCallback(
            model=model,
            dev_samples=dev_samples,
            dev_labels=dev_labels,
            eval_steps=args.eval_steps,
            early_stopping_patience=args.early_stopping_patience,
            model_dir=args.model_dir,
        )
        callbacks.append(dev_callback)
        if accelerator.is_main_process:
            print(f"Dev evaluation enabled: every {args.eval_steps} steps, patience={args.early_stopping_patience}")

    # Training arguments
    training_args = SentenceTransformerTrainingArguments(
        output_dir=args.model_dir,
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        fp16=torch.cuda.is_available(),
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
    )

    # Train
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=train_loss,
        callbacks=callbacks,
    )

    trainer.train()

    # Wait for all processes to finish training
    accelerator.wait_for_everyone()

    # Save model (only on main process)
    if accelerator.is_main_process:
        # If we have a best model from dev eval, use that
        best_model_dir = os.path.join(args.model_dir, "best_model")
        if dev_callback and os.path.exists(best_model_dir):
            print(f"Using best model from step {dev_callback.best_step}")
            # Copy best model files to model_dir
            for item in os.listdir(best_model_dir):
                src = os.path.join(best_model_dir, item)
                dst = os.path.join(args.model_dir, item)
                if os.path.isdir(src):
                    if os.path.exists(dst):
                        shutil.rmtree(dst)
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)
        else:
            model.save_pretrained(args.model_dir)
        print(f"Model saved to: {args.model_dir}")

        # Save eval report
        if dev_callback:
            report = dev_callback.get_report()
            report_path = os.path.join(args.model_dir, "eval_report.json")
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"Eval report saved to: {report_path}")
            print(f"Best NDCG@10: {dev_callback.best_ndcg:.4f} at step {dev_callback.best_step}")
            if dev_callback.baseline_metrics:
                print(f"Baseline NDCG@10: {dev_callback.baseline_metrics['ndcg@10']:.4f}")

        # Remove checkpoints and training artifacts from model dir
        for item in os.listdir(args.model_dir):
            item_path = os.path.join(args.model_dir, item)
            if os.path.isdir(item_path) and item.startswith("checkpoint-"):
                shutil.rmtree(item_path)
                print(f"Removed checkpoint: {item}")
            elif os.path.isdir(item_path) and item == "best_model":
                shutil.rmtree(item_path)
                print(f"Removed best_model temp dir")
            elif item in ("training_args.bin", "optimizer.pt", "scheduler.pt"):
                os.remove(item_path)
                print(f"Removed {item}")

        # Create model tarball for SageMaker (exclude checkpoints and unnecessary files)
        model_tar_path = os.path.join(args.model_dir, "model.tar.gz")
        exclude_patterns = {"model.tar.gz", "checkpoint-", "best_model", "runs", "training_args.bin", "optimizer.pt", "scheduler.pt"}
        with tarfile.open(model_tar_path, "w:gz") as tar:
            for item in os.listdir(args.model_dir):
                if item == "model.tar.gz":
                    continue
                if any(item.startswith(p) for p in exclude_patterns):
                    print(f"Skipping {item} (not needed for inference)")
                    continue
                tar.add(os.path.join(args.model_dir, item), arcname=item)
        print(f"Model tarball created: {model_tar_path}")


if __name__ == "__main__":
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    # If multiple GPUs available and not already in distributed mode, launch with torchrun
    if num_gpus > 1 and "LOCAL_RANK" not in os.environ:
        print(f"Detected {num_gpus} GPUs, launching with DDP via torchrun...")
        cmd = [
            sys.executable, "-m", "torch.distributed.run",
            "--nproc_per_node", str(num_gpus),
            "--master_port", "29500",
            sys.argv[0]
        ] + sys.argv[1:]
        sys.exit(subprocess.call(cmd))
    else:
        main()
