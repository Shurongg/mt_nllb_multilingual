"""NLLB-specific utility placeholders.

TODO:
- Centralize NLLB language codes.
- Add tokenizer setup helpers.
- Add generation configuration helpers.
- Add validation that only supported experiment modes are used.
"""

JAVANESE = "jav_Latn"
INDONESIAN = "ind_Latn"
ENGLISH = "eng_Latn"

SUPPORTED_MODES = ("zero_shot", "java_only", "joint_balanced")

