"""
Shared constants and utilities for the WX-AFD pipeline.

Heavy imports (torch, transformers, rouge_score, bert_score) are deferred
to inside the functions that need them so lightweight scripts like
03_build_dataset.py don't pay for GPU library initialization.
"""

SYSTEM_PROMPT = """\
You are an expert NWS meteorologist at the Louisville, Kentucky Weather \
Forecast Office (WFO LMK). Write an Area Forecast Discussion (AFD) based \
on the provided weather model data.

Follow standard NWS formatting:
- Begin with .KEY MESSAGES or .WHAT HAS CHANGED
- Include .SYNOPSIS, .SHORT TERM, .LONG TERM, .AVIATION sections
- Reference specific model data (temps, winds, precip, pressure patterns)
- Discuss confidence levels and model agreement
- Use NWS abbreviations and meteorological terminology
- Reference Ohio Valley, Bluegrass, and local geographic features"""

REQUIRED_SECTIONS = [".synopsis", ".short term", ".long term", ".aviation"]


def load_jsonl(path):
    """Load a JSONL file into a list of dicts."""
    import json
    with open(path) as f:
        return [json.loads(line) for line in f]


def load_model(path):
    """Load a HuggingFace causal LM and its tokenizer in bfloat16."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def generate_afd(model, tokenizer, weather_input, max_new_tokens=2048,
                 temperature=0.7, top_p=0.9):
    """Generate an AFD from weather input using the shared system prompt."""
    import torch

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": weather_input},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    pad_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
        )
    generated = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def compute_rouge(predictions, references):
    """Compute average ROUGE-1/2/L F-measure scores."""
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"],
                                      use_stemmer=True)
    scores = {"rouge1": [], "rouge2": [], "rougeL": []}
    for pred, ref in zip(predictions, references):
        result = scorer.score(ref, pred)
        for key in scores:
            scores[key].append(result[key].fmeasure)
    return {k: sum(v) / len(v) for k, v in scores.items()}


def compute_bertscore(predictions, references):
    """Compute mean BERTScore F1. Imports lazily to manage GPU memory."""
    from bert_score import score as bert_score
    P, R, F1 = bert_score(predictions, references, lang="en", verbose=True)
    return float(F1.mean())


def format_compliance(text):
    """Check how many required AFD sections are present."""
    low = text.lower()
    found = [sec for sec in REQUIRED_SECTIONS if sec in low]
    return {
        "sections_found": found,
        "sections_missing": [s for s in REQUIRED_SECTIONS if s not in low],
        "compliance_score": len(found) / len(REQUIRED_SECTIONS),
    }
