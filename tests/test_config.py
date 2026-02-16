"""
Pre-flight validation for configs/wx-afd-dora.yml.

Catches common axolotl config errors locally before wasting Derecho GPU queue time.
Requires only pytest + pyyaml — no axolotl, torch, or GPU needed.
"""
import yaml
import pytest
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "wx-afd-dora.yml"


@pytest.fixture(scope="module")
def cfg():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ── 1. Required top-level fields ──────────────────────────────────────────────

REQUIRED_FIELDS = [
    "base_model",
    "datasets",
    "learning_rate",
    "num_epochs",
    "output_dir",
    "sequence_len",
]


def test_yaml_loads_and_has_required_fields(cfg):
    assert cfg is not None, "YAML failed to load"
    for field in REQUIRED_FIELDS:
        assert field in cfg, f"Missing required field: {field}"


# ── 2. train_on_eos must be a string, not a bool ─────────────────────────────

def test_train_on_eos_is_string_not_bool(cfg):
    valid_values = {"all", "turn", "last"}
    for i, ds in enumerate(cfg["datasets"]):
        if "train_on_eos" in ds:
            val = ds["train_on_eos"]
            assert not isinstance(val, bool), (
                f"datasets[{i}].train_on_eos is bool ({val}); "
                f"axolotl 0.14+ requires one of {valid_values}"
            )
            assert val in valid_values, (
                f"datasets[{i}].train_on_eos={val!r}; expected one of {valid_values}"
            )


# ── 3. batch_size and gradient_accumulation_steps are mutually exclusive ──────

def test_batch_size_not_both(cfg):
    has_batch = "batch_size" in cfg
    has_grad_accum = "gradient_accumulation_steps" in cfg
    assert not (has_batch and has_grad_accum), (
        "Set batch_size OR gradient_accumulation_steps, not both. "
        "axolotl computes grad_accum = batch_size / micro_batch_size automatically."
    )


# ── 4. batch_size divisible by micro_batch_size ──────────────────────────────

def test_batch_size_divisibility(cfg):
    has_batch = "batch_size" in cfg
    has_grad_accum = "gradient_accumulation_steps" in cfg
    assert has_batch or has_grad_accum, (
        "Must set either batch_size or gradient_accumulation_steps"
    )
    if has_batch:
        assert "micro_batch_size" in cfg, "micro_batch_size is required when batch_size is set"
        bs = cfg["batch_size"]
        mbs = cfg["micro_batch_size"]
        assert bs % mbs == 0, (
            f"batch_size ({bs}) must be divisible by micro_batch_size ({mbs})"
        )
    if has_grad_accum:
        ga = cfg["gradient_accumulation_steps"]
        assert isinstance(ga, int) and ga > 0, (
            f"gradient_accumulation_steps must be a positive integer, got {ga!r}"
        )


# ── 5. Boolean fields are actually bools ─────────────────────────────────────

BOOLEAN_FIELDS = [
    "bf16",
    "tf32",
    "gradient_checkpointing",
    "sample_packing",
    "flash_attention",
    "trust_remote_code",
    "load_best_model_at_end",
    "lora_target_linear",
    "peft_use_rslora",
    "peft_use_dora",
]


def test_boolean_fields_are_bool(cfg):
    for field in BOOLEAN_FIELDS:
        if field in cfg:
            val = cfg[field]
            assert isinstance(val, bool), (
                f"{field}={val!r} (type {type(val).__name__}); expected bool"
            )


# ── 6. Numeric ranges ────────────────────────────────────────────────────────

def test_numeric_ranges(cfg):
    lr = cfg["learning_rate"]
    assert isinstance(lr, float), f"learning_rate should be float, got {type(lr).__name__}"
    assert 1e-7 < lr < 1e-1, f"learning_rate={lr} looks wrong (expected ~1e-5 to 1e-3)"

    epochs = cfg["num_epochs"]
    assert isinstance(epochs, int), f"num_epochs should be int, got {type(epochs).__name__}"
    assert 1 <= epochs <= 100, f"num_epochs={epochs} out of reasonable range"

    seq_len = cfg["sequence_len"]
    assert isinstance(seq_len, int), f"sequence_len should be int, got {type(seq_len).__name__}"
    assert 128 <= seq_len <= 131072, f"sequence_len={seq_len} out of range"


# ── 7. Dataset structure ─────────────────────────────────────────────────────

def test_dataset_structure(cfg):
    assert isinstance(cfg["datasets"], list), "datasets must be a list"
    assert len(cfg["datasets"]) > 0, "datasets list is empty"

    for i, ds in enumerate(cfg["datasets"]):
        for key in ("path", "type"):
            assert key in ds, f"datasets[{i}] missing required key: {key}"
        if ds.get("type") == "chat_template":
            assert "roles_to_train" in ds, (
                f"datasets[{i}] uses chat_template but missing roles_to_train"
            )


# ── 8. field_messages must be set on chat_template datasets ────────────────

def _all_chat_template_datasets(cfg):
    """Yield (label, dataset_dict) for every chat_template dataset."""
    for i, ds in enumerate(cfg.get("datasets", [])):
        if ds.get("type") == "chat_template":
            yield f"datasets[{i}]", ds
    for i, ds in enumerate(cfg.get("test_datasets", [])):
        if ds.get("type") == "chat_template":
            yield f"test_datasets[{i}]", ds


def test_field_messages_set_on_chat_template_datasets(cfg):
    """Axolotl defaults to 'conversations' but our JSONL uses 'messages'.
    Omitting field_messages causes ValueError: Messages is null."""
    found_any = False
    for label, ds in _all_chat_template_datasets(cfg):
        found_any = True
        assert "field_messages" in ds, (
            f"{label} uses chat_template but missing field_messages — "
            "Axolotl will look for 'conversations' instead of 'messages'"
        )
    assert found_any, "No chat_template datasets found — config may be misconfigured"


# ── 8. EOS and PAD tokens must differ ────────────────────────────────────────

def test_special_tokens_eos_differs_from_pad(cfg):
    assert "special_tokens" in cfg, "special_tokens section is required"
    tokens = cfg["special_tokens"]
    assert "eos_token" in tokens, "special_tokens.eos_token is required"
    assert "pad_token" in tokens, "special_tokens.pad_token is required"
    eos = tokens["eos_token"]
    pad = tokens["pad_token"]
    assert eos != pad, (
        f"eos_token and pad_token are both {eos!r} — "
        "model will never learn to stop generating"
    )


# ── 9. eot_tokens must be a list of strings ─────────────────────────────────

def test_eot_tokens_is_list_of_strings(cfg):
    assert "eot_tokens" in cfg, "eot_tokens is required for chat models"
    val = cfg["eot_tokens"]
    assert isinstance(val, list), (
        f"eot_tokens must be a list, got {type(val).__name__}: {val!r}"
    )
    for i, tok in enumerate(val):
        assert isinstance(tok, str), (
            f"eot_tokens[{i}]={tok!r} is {type(tok).__name__}, expected str"
        )


# ── 10. Mutually exclusive field pairs ───────────────────────────────────────

MUTUALLY_EXCLUSIVE_PAIRS = [
    ("warmup_steps", "warmup_ratio"),
    ("eval_steps", "evals_per_epoch"),
    ("save_steps", "saves_per_epoch"),
]


def test_mutually_exclusive_fields(cfg):
    for field_a, field_b in MUTUALLY_EXCLUSIVE_PAIRS:
        both_set = field_a in cfg and field_b in cfg
        assert not both_set, (
            f"{field_a} and {field_b} are mutually exclusive — set one or the other"
        )
