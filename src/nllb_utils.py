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
