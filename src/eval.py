"""Evaluation metrics using decoder tokenizer for phoneme decoding."""

import numpy as np
import jiwer
from typing import List

try:
    import wandb
except ImportError:
    wandb = None


def calculate_wer(references: List[str], hypotheses: List[str]) -> float:
    """Calculate Word Error Rate."""
    return jiwer.wer(references, hypotheses)


def calculate_cer(references: List[str], hypotheses: List[str]) -> float:
    """Calculate Character Error Rate."""
    return jiwer.cer(references, hypotheses)


def calculate_exact_match(references: List[str], hypotheses: List[str]) -> float:
    """Calculate percentage of exact matches."""
    matches = sum(r == h for r, h in zip(references, hypotheses))
    return matches / len(references) if references else 0.0


def log_vocab_range(predictions, vocab_size):
    """Check for out-of-vocabulary predictions."""
    min_id, max_id = predictions.min(), predictions.max()
    if max_id >= vocab_size:
        oov_count = (predictions >= vocab_size).sum()
        print(f"WARNING: {oov_count} out-of-vocab predictions (range: [{min_id}, {max_id}], vocab: {vocab_size})")


def log_eval_predictions(decoded_preds: List[str], decoded_labels: List[str], num_samples: int = 5):
    """Log sample predictions during evaluation."""
    print(f"\n{'='*60}\nEval Samples ({min(num_samples, len(decoded_preds))} of {len(decoded_preds)})\n{'='*60}")

    for i in range(min(num_samples, len(decoded_preds))):
        match = decoded_preds[i] == decoded_labels[i]
        print(f"{'✓' if match else '✗'} [{i+1}] {decoded_labels[i]} → {decoded_preds[i]}")

    print('='*60 + '\n')

    if wandb and wandb.run:
        data = [[i, decoded_labels[i], decoded_preds[i], "✓" if decoded_preds[i] == decoded_labels[i] else "✗"]
                for i in range(min(num_samples, len(decoded_preds)))]
        wandb.log({"eval_examples": wandb.Table(columns=["Index", "Reference", "Predicted", "Match"], data=data)})


def create_compute_metrics(decoder_tokenizer):
    """Create metrics function using decoder tokenizer for phoneme decoding."""
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        labels = np.where(labels != -100, labels, decoder_tokenizer.pad_token_id)

        vocab_size = len(decoder_tokenizer)
        log_vocab_range(predictions, vocab_size)
        predictions = np.clip(predictions, 0, vocab_size - 1)

        decoded_preds = decoder_tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = decoder_tokenizer.batch_decode(labels, skip_special_tokens=True)

        log_eval_predictions(decoded_preds, decoded_labels, num_samples=5)

        wer = calculate_wer(decoded_labels, decoded_preds)
        cer = calculate_cer(decoded_labels, decoded_preds)
        exact_match = calculate_exact_match(decoded_labels, decoded_preds)

        return {"wer": wer, "cer": cer, "exact_match": exact_match}

    return compute_metrics
