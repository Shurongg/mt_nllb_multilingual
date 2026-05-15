"""NLLB-specific constants and helpers."""

JAVANESE = "jav_Latn"
INDONESIAN = "ind_Latn"
ENGLISH = "eng_Latn"

SUPPORTED_MODES = ("zero_shot", "java_only", "joint_balanced")


def get_forced_bos_token_id(tokenizer, target_lang: str) -> int:
    """Return the NLLB forced BOS token id for the target language."""
    token_id = tokenizer.convert_tokens_to_ids(target_lang)
    if token_id is None or token_id == tokenizer.unk_token_id:
        raise ValueError(
            f"Target language token {target_lang!r} was not found in the tokenizer."
        )
    return token_id


def get_device_summary(torch_module) -> dict:
    """Return a compact summary of the available torch device backend."""
    cuda_available = torch_module.cuda.is_available()
    mps_available = bool(
        hasattr(torch_module.backends, "mps")
        and torch_module.backends.mps.is_available()
    )
    summary = {
        "cuda_available": cuda_available,
        "mps_available": mps_available,
        "device": "cuda" if cuda_available else "mps" if mps_available else "cpu",
    }
    if cuda_available:
        summary["cuda_device_count"] = torch_module.cuda.device_count()
        summary["cuda_device_name"] = torch_module.cuda.get_device_name(0)
    return summary


def resolve_fp16(fp16_requested: bool, torch_module) -> bool:
    """Use fp16 only on CUDA; disable it safely on CPU/MPS."""
    if not fp16_requested:
        return False
    if torch_module.cuda.is_available():
        return True
    print("Warning: fp16=true was requested, but CUDA is not available. Disabling fp16.")
    return False
