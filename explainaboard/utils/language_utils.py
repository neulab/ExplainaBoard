"""All language codes in the ISO 639-3 Chinese macro language family
 (https://iso639-3.sil.org/code/zho)"""

from __future__ import annotations

CHINESE_MACRO_FAMILY = {
    'zh',
    'zho',
    'cdo',
    'cjy',
    'cmn',
    'cpx',
    'czh',
    'czo',
    'gan',
    'hak',
    'hsn',
    'lzh',
    'mnp',
    'nan',
    'wuu',
    'yue',
    'cnp',
    'csp',
}


def is_chinese_lang_code(lang_code: str | None) -> bool:
    return lang_code in CHINESE_MACRO_FAMILY


def is_japanese_lang_code(lang_code: str | None) -> bool:
    return lang_code in {'ja', 'jpn'}
