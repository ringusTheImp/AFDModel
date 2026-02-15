#!/usr/bin/env python3
"""
Evaluate a fine-tuned WX-AFD model against reference AFDs.

Metrics: ROUGE-1/2/L, BERTScore F1, format compliance.
Optional: AlignScore (--alignscore flag).

Usage:
    python evaluate.py --model /path/to/merged/model --val_data /path/to/val.jsonl
    python evaluate.py --model /path/to/merged/model --val_data /path/to/val.jsonl --num_examples 10
    python evaluate.py --model Qwen/Qwen3-4B-Instruct-2507 --val_data /path/to/val.jsonl --tag zero-shot
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from rouge_score import rouge_scorer
from tqdm import tqdm

from wx_afd import (
    SYSTEM_PROMPT, REQUIRED_SECTIONS, load_model, generate_afd,
    compute_rouge, compute_bertscore, format_compliance,
)


def load_val_data(path, num_examples=None):
    """Load validation data in messages format."""
    examples = []
    with open(path) as f:
        for line in f:
            ex = json.loads(line)
            messages = ex["messages"]
            weather_input = messages[1]["content"]
            reference_afd = messages[2]["content"]
            examples.append({"input": weather_input, "reference": reference_afd})
    if num_examples is not None:
        examples = examples[:num_examples]
    return examples


def main():
    parser = argparse.ArgumentParser(description="Evaluate WX-AFD model")
    parser.add_argument("--model", required=True,
                        help="Path to merged model or HF model ID")
    parser.add_argument("--val_data", required=True,
                        help="Path to val.jsonl in messages format")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory for results (default: eval/<tag>/)")
    parser.add_argument("--tag", default="finetuned",
                        help="Tag for this eval run (e.g., 'finetuned', 'zero-shot')")
    parser.add_argument("--num_examples", type=int, default=None,
                        help="Evaluate only first N examples")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--alignscore", action="store_true",
                        help="Also compute AlignScore (requires alignscore package)")
    args = parser.parse_args()

    # Output dirs
    if args.output_dir is None:
        base = Path(args.val_data).parent.parent / "eval" / args.tag
    else:
        base = Path(args.output_dir)
    gen_dir = base / "generated"
    scores_dir = base / "scores"
    gen_dir.mkdir(parents=True, exist_ok=True)
    scores_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading validation data from {args.val_data}")
    examples = load_val_data(args.val_data, args.num_examples)
    print(f"  {len(examples)} examples loaded")

    # Load model
    print(f"Loading model: {args.model}")
    model, tokenizer = load_model(args.model)

    # Generate
    predictions = []
    references = []
    per_example = []

    print("Generating AFDs...")
    for i, ex in enumerate(tqdm(examples)):
        pred = generate_afd(model, tokenizer, ex["input"],
                            max_new_tokens=args.max_new_tokens,
                            temperature=args.temperature,
                            top_p=args.top_p)
        ref = ex["reference"]
        predictions.append(pred)
        references.append(ref)

        # Save individual generation
        with open(gen_dir / f"example_{i:04d}.txt", "w") as f:
            f.write(pred)

        per_example.append({
            "index": i,
            "pred_len": len(pred),
            "ref_len": len(ref),
            "format": format_compliance(pred),
        })

    # Free model memory before BERTScore
    del model
    torch.cuda.empty_cache()

    # ROUGE
    print("Computing ROUGE scores...")
    rouge_scores = compute_rouge(predictions, references)
    print(f"  ROUGE-1: {rouge_scores['rouge1']:.4f}")
    print(f"  ROUGE-2: {rouge_scores['rouge2']:.4f}")
    print(f"  ROUGE-L: {rouge_scores['rougeL']:.4f}")

    # Per-example ROUGE
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"],
                                      use_stemmer=True)
    for i, (pred, ref) in enumerate(zip(predictions, references)):
        result = scorer.score(ref, pred)
        per_example[i]["rouge1"] = result["rouge1"].fmeasure
        per_example[i]["rouge2"] = result["rouge2"].fmeasure
        per_example[i]["rougeL"] = result["rougeL"].fmeasure

    # BERTScore
    print("Computing BERTScore...")
    bertscore_f1 = compute_bertscore(predictions, references)
    print(f"  BERTScore F1: {bertscore_f1:.4f}")

    # Format compliance
    compliance_scores = [format_compliance(p)["compliance_score"]
                         for p in predictions]
    avg_compliance = sum(compliance_scores) / len(compliance_scores)
    print(f"  Format compliance: {avg_compliance:.2%}")

    for i, cs in enumerate(compliance_scores):
        per_example[i]["compliance_score"] = cs

    # AlignScore (optional)
    alignscore_mean = None
    if args.alignscore:
        try:
            from alignscore import AlignScorer
            print("Computing AlignScore...")
            align_scorer = AlignScorer(model="roberta-large", batch_size=16,
                                       device="cuda")
            inputs_for_align = [ex["input"] for ex in examples]
            align_scores = align_scorer.score(contexts=inputs_for_align,
                                              claims=predictions)
            alignscore_mean = sum(align_scores) / len(align_scores)
            print(f"  AlignScore: {alignscore_mean:.4f}")
            for i, s in enumerate(align_scores):
                per_example[i]["alignscore"] = s
        except ImportError:
            print("  AlignScore not installed, skipping (pip install alignscore)")

    # Save metrics
    metrics = {
        "model": args.model,
        "tag": args.tag,
        "num_examples": len(examples),
        "rouge1": rouge_scores["rouge1"],
        "rouge2": rouge_scores["rouge2"],
        "rougeL": rouge_scores["rougeL"],
        "bertscore_f1": bertscore_f1,
        "format_compliance": avg_compliance,
    }
    if alignscore_mean is not None:
        metrics["alignscore"] = alignscore_mean

    with open(scores_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {scores_dir / 'metrics.json'}")

    # Save per-example scores
    with open(scores_dir / "per_example.jsonl", "w") as f:
        for ex in per_example:
            f.write(json.dumps(ex) + "\n")
    print(f"Per-example scores saved to {scores_dir / 'per_example.jsonl'}")

    # Summary
    print("\n" + "=" * 60)
    print(f"  Evaluation Summary: {args.tag}")
    print("=" * 60)
    print(f"  ROUGE-1:          {metrics['rouge1']:.4f}")
    print(f"  ROUGE-2:          {metrics['rouge2']:.4f}")
    print(f"  ROUGE-L:          {metrics['rougeL']:.4f}")
    print(f"  BERTScore F1:     {metrics['bertscore_f1']:.4f}")
    print(f"  Format Compliance: {metrics['format_compliance']:.2%}")
    if alignscore_mean is not None:
        print(f"  AlignScore:       {metrics['alignscore']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
