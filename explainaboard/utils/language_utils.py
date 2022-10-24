"""Information of language codes."""

from __future__ import annotations

"""All language codes in the ISO 639-3 (https://iso639-3.sil.org/code/zho)."""
CHINESE_MACRO_FAMILY = {
    "zh",
    "zho",
    "cdo",
    "cjy",
    "cmn",
    "cpx",
    "czh",
    "czo",
    "gan",
    "hak",
    "hsn",
    "lzh",
    "mnp",
    "nan",
    "wuu",
    "yue",
    "cnp",
    "csp",
}


def is_chinese_lang_code(lang_code: str | None) -> bool:
    """Judge whether the language is Chinese."""
    return lang_code in CHINESE_MACRO_FAMILY


def is_japanese_lang_code(lang_code: str | None) -> bool:
    """Judge whether if the language is Japanese."""
    return lang_code in {"ja", "jpn"}
