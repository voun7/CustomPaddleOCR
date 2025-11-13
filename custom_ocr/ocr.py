def get_ocr_model_names(lang, ppocr_version):
    LATIN_LANGS = [
        "af",
        "az",
        "bs",
        "cs",
        "cy",
        "da",
        "de",
        "es",
        "et",
        "fr",
        "ga",
        "hr",
        "hu",
        "id",
        "is",
        "it",
        "ku",
        "la",
        "lt",
        "lv",
        "mi",
        "ms",
        "mt",
        "nl",
        "no",
        "oc",
        "pi",
        "pl",
        "pt",
        "ro",
        "rs_latin",
        "sk",
        "sl",
        "sq",
        "sv",
        "sw",
        "tl",
        "tr",
        "uz",
        "vi",
        "french",
        "german",
        "fi",
        "eu",
        "gl",
        "lb",
        "rm",
        "ca",
        "qu",
    ]
    ARABIC_LANGS = ["ar", "fa", "ug", "ur", "ps", "ku", "sd", "bal"]
    ESLAV_LANGS = ["ru", "be", "uk"]
    CYRILLIC_LANGS = [
        "ru",
        "rs_cyrillic",
        "be",
        "bg",
        "uk",
        "mn",
        "abq",
        "ady",
        "kbd",
        "ava",
        "dar",
        "inh",
        "che",
        "lbe",
        "lez",
        "tab",
        "kk",
        "ky",
        "tg",
        "mk",
        "tt",
        "cv",
        "ba",
        "mhr",
        "mo",
        "udm",
        "kv",
        "os",
        "bua",
        "xal",
        "tyv",
        "sah",
        "kaa",
    ]
    DEVANAGARI_LANGS = [
        "hi",
        "mr",
        "ne",
        "bh",
        "mai",
        "ang",
        "bho",
        "mah",
        "sck",
        "new",
        "gom",
        "sa",
        "bgc",
    ]
    SPECIFIC_LANGS = [
        "ch",
        "en",
        "korean",
        "japan",
        "chinese_cht",
        "te",
        "ka",
        "ta",
    ]

    if lang is None:
        lang = "ch"

    if ppocr_version is None:
        if (lang in ["ch", "chinese_cht", "en", "japan", "korean", "th", "el", "te", "ta"]
                + LATIN_LANGS
                + ESLAV_LANGS
                + ARABIC_LANGS
                + CYRILLIC_LANGS
                + DEVANAGARI_LANGS
        ):
            ppocr_version = "PP-OCRv5"
        elif lang in (SPECIFIC_LANGS):
            ppocr_version = "PP-OCRv3"
        else:
            # Unknown language specified
            return None, None

    if ppocr_version == "PP-OCRv5":
        rec_lang, rec_model_name = None, None
        if lang in ("ch", "chinese_cht", "japan"):
            rec_model_name = "PP-OCRv5_server_rec"
        elif lang == "en":
            rec_model_name = "en_PP-OCRv5_mobile_rec"
        elif lang in LATIN_LANGS:
            rec_lang = "latin"
        elif lang in ESLAV_LANGS:
            rec_lang = "eslav"
        elif lang in ARABIC_LANGS:
            rec_lang = "arabic"
        elif lang in CYRILLIC_LANGS:
            rec_lang = "cyrillic"
        elif lang in DEVANAGARI_LANGS:
            rec_lang = "devanagari"
        elif lang == "korean":
            rec_lang = "korean"
        elif lang == "th":
            rec_lang = "th"
        elif lang == "el":
            rec_lang = "el"
        elif lang == "te":
            rec_lang = "te"
        elif lang == "ta":
            rec_lang = "ta"

        if rec_lang is not None:
            rec_model_name = f"{rec_lang}_PP-OCRv5_mobile_rec"
        return "PP-OCRv5_server_det", rec_model_name

    elif ppocr_version == "PP-OCRv4":
        if lang == "ch":
            return "PP-OCRv4_mobile_det", "PP-OCRv4_mobile_rec"
        elif lang == "en":
            return "PP-OCRv4_mobile_det", "en_PP-OCRv4_mobile_rec"
        else:
            return None, None
    else:
        # PP-OCRv3
        rec_lang = None
        if lang in LATIN_LANGS:
            rec_lang = "latin"
        elif lang in ARABIC_LANGS:
            rec_lang = "arabic"
        elif lang in CYRILLIC_LANGS:
            rec_lang = "cyrillic"
        elif lang in DEVANAGARI_LANGS:
            rec_lang = "devanagari"
        else:
            if lang in SPECIFIC_LANGS:
                rec_lang = lang

        rec_model_name = None
        if rec_lang == "ch":
            rec_model_name = "PP-OCRv3_mobile_rec"
        elif rec_lang is not None:
            rec_model_name = f"{rec_lang}_PP-OCRv3_mobile_rec"
        return "PP-OCRv3_mobile_det", rec_model_name


class CustomPaddleOCR:
    ...
