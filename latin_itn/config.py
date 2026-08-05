"""
Configuration constants and label maps for Latin ITN.
"""

DEFAULT_MODEL_NAME = "njand/latin-asr-postprocessor"

# 14-Tag ITN Schema (Casing + Trailing Punctuation)
TAG_LIST = [
    "LOWER_NONE",
    "LOWER_COMMA",
    "LOWER_PERIOD",
    "LOWER_QUESTION",
    "LOWER_EXCLAMATION",
    "LOWER_SEMICOLON",
    "LOWER_COLON",
    "TITLE_NONE",
    "TITLE_COMMA",
    "TITLE_PERIOD",
    "TITLE_QUESTION",
    "TITLE_EXCLAMATION",
    "TITLE_SEMICOLON",
    "TITLE_COLON",
]

LABEL2ID = {tag: i for i, tag in enumerate(TAG_LIST)}
ID2LABEL = {i: tag for i, tag in enumerate(TAG_LIST)}

PUNCT_MAP = {
    "NONE": "",
    "COMMA": ",",
    "PERIOD": ".",
    "QUESTION": "?",
    "EXCLAMATION": "!",
    "SEMICOLON": ";",
    "COLON": ":",
}
