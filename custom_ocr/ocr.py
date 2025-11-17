import warnings
from pathlib import Path

import yaml

from .inference.image_classification.predictor import ClasPredictor
from .inference.text_detection.predictor import TextDetPredictor
from .inference.text_recognition.predictor import TextRecPredictor
from .pipeline.ocr import OCRPipeline


class TextDetection:
    def __init__(self, model_save_dir: str, model_name: str = "PP-OCRv5_server_det", **kwargs):
        self.predictor = TextDetPredictor(model_save_dir=model_save_dir, model_name=model_name, **kwargs)

    def predict_iter(self, input_):
        return self.predictor.predict(input_)

    def predict(self, input_):
        result = list(self.predict_iter(input_))
        return result


class TextRecognition:
    def __init__(self, model_save_dir: str, model_name: str = "PP-OCRv5_server_rec", **kwargs):
        self.predictor = TextRecPredictor(model_save_dir=model_save_dir, model_name=model_name, **kwargs)

    def predict_iter(self, input_):
        return self.predictor.predict(input_)

    def predict(self, input_):
        result = list(self.predict_iter(input_))
        return result


class TextLineOrientationClassification:
    def __init__(self, model_save_dir: str, model_name: str = "PP-LCNet_x0_25_textline_ori", **kwargs):
        self.predictor = ClasPredictor(model_save_dir=model_save_dir, model_name=model_name, **kwargs)

    def predict_iter(self, input_):
        return self.predictor.predict(input_)

    def predict(self, input_):
        result = list(self.predict_iter(input_))
        return result


def _merge_dicts(d1, d2):
    res = d1.copy()
    for k, v in d2.items():
        if k in res and isinstance(res[k], dict) and isinstance(v, dict):
            res[k] = _merge_dicts(res[k], v)
        else:
            res[k] = v
    return res


def create_config_from_structure(structure, *, unset=None, config=None):
    if config is None:
        config = {}
    for k, v in structure.items():
        if v is unset:
            continue
        idx = k.find(".")
        if idx == -1:
            config[k] = v
        else:
            sk = k[:idx]
            if sk not in config:
                config[sk] = {}
            create_config_from_structure({k[idx + 1:]: v}, config=config[sk])
    return config


class CustomPaddleOCR:
    _supported_ocr_versions = ["PP-OCRv3", "PP-OCRv4", "PP-OCRv5"]

    def __init__(
            self,
            model_save_dir: str,
            use_mobile_model=False,  # Only works when 'lang' is given
            text_detection_model_name=None,
            text_detection_model_dir=None,
            textline_orientation_model_name=None,
            textline_orientation_model_dir=None,
            textline_orientation_batch_size=None,
            text_recognition_model_name=None,
            text_recognition_model_dir=None,
            text_recognition_batch_size=None,
            use_textline_orientation=None,
            text_det_limit_side_len=None,
            text_det_limit_type=None,
            text_det_thresh=None,
            text_det_box_thresh=None,
            text_det_unclip_ratio=None,
            text_det_input_shape=None,
            text_rec_score_thresh=None,
            return_word_box=None,
            text_rec_input_shape=None,
            lang=None,
            ocr_version=None,
            **kwargs,
    ):
        if ocr_version is not None and ocr_version not in self._supported_ocr_versions:
            raise ValueError(f"Invalid OCR version: {ocr_version}. "
                             f"Supported values are {self._supported_ocr_versions}.")

        if all(
                map(
                    lambda p: p is None,
                    (
                            text_detection_model_name,
                            text_detection_model_dir,
                            text_recognition_model_name,
                            text_recognition_model_dir,
                    ),
                )
        ):
            if lang is not None or ocr_version is not None:
                det_model_name, rec_model_name = self._get_ocr_model_names(lang, ocr_version, use_mobile_model)
                if det_model_name is None or rec_model_name is None:
                    raise ValueError(f"No models are available for the language {repr(lang)} and "
                                     f"OCR version {repr(ocr_version)}.")
                text_detection_model_name = det_model_name
                text_recognition_model_name = rec_model_name
        else:
            if lang is not None or ocr_version is not None:
                warnings.warn("`lang` and `ocr_version` will be ignored when model names or "
                              "model directories are not `None`.", stacklevel=2)

        params = {
            "doc_orientation_classify_model_name": None,
            "doc_orientation_classify_model_dir": None,
            "doc_unwarping_model_name": None,
            "doc_unwarping_model_dir": None,
            "text_detection_model_name": text_detection_model_name,
            "text_detection_model_dir": text_detection_model_dir,
            "textline_orientation_model_name": textline_orientation_model_name,
            "textline_orientation_model_dir": textline_orientation_model_dir,
            "textline_orientation_batch_size": textline_orientation_batch_size,
            "text_recognition_model_name": text_recognition_model_name,
            "text_recognition_model_dir": text_recognition_model_dir,
            "text_recognition_batch_size": text_recognition_batch_size,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": use_textline_orientation,
            "text_det_limit_side_len": text_det_limit_side_len,
            "text_det_limit_type": text_det_limit_type,
            "text_det_thresh": text_det_thresh,
            "text_det_box_thresh": text_det_box_thresh,
            "text_det_unclip_ratio": text_det_unclip_ratio,
            "text_det_input_shape": text_det_input_shape,
            "text_rec_score_thresh": text_rec_score_thresh,
            "return_word_box": return_word_box,
            "text_rec_input_shape": text_rec_input_shape,
        }

        kwargs["model_save_dir"] = model_save_dir
        self._params = params
        pipeline_config = self._get_merged_config()
        self.ocr_pipline = OCRPipeline(pipeline_config, **kwargs)

    def _get_config_overrides(self):
        structure = {
            "SubPipelines.DocPreprocessor.SubModules.DocOrientationClassify.model_name": self._params[
                "doc_orientation_classify_model_name"],
            "SubPipelines.DocPreprocessor.SubModules.DocOrientationClassify.model_dir": self._params[
                "doc_orientation_classify_model_dir"],
            "SubPipelines.DocPreprocessor.SubModules.DocUnwarping.model_name": self._params[
                "doc_unwarping_model_name"],
            "SubPipelines.DocPreprocessor.SubModules.DocUnwarping.model_dir": self._params["doc_unwarping_model_dir"],
            "SubModules.TextDetection.model_name": self._params["text_detection_model_name"],
            "SubModules.TextDetection.model_dir": self._params["text_detection_model_dir"],
            "SubModules.TextLineOrientation.model_name": self._params["textline_orientation_model_name"],
            "SubModules.TextLineOrientation.model_dir": self._params["textline_orientation_model_dir"],
            "SubModules.TextLineOrientation.batch_size": self._params["textline_orientation_batch_size"],
            "SubModules.TextRecognition.model_name": self._params["text_recognition_model_name"],
            "SubModules.TextRecognition.model_dir": self._params["text_recognition_model_dir"],
            "SubModules.TextRecognition.batch_size": self._params["text_recognition_batch_size"],
            "SubPipelines.DocPreprocessor.use_doc_orientation_classify": self._params["use_doc_orientation_classify"],
            "SubPipelines.DocPreprocessor.use_doc_unwarping": self._params["use_doc_unwarping"],
            "use_doc_preprocessor": self._params["use_doc_orientation_classify"] or self._params["use_doc_unwarping"],
            "use_textline_orientation": self._params["use_textline_orientation"],
            "SubModules.TextDetection.limit_side_len": self._params["text_det_limit_side_len"],
            "SubModules.TextDetection.limit_type": self._params["text_det_limit_type"],
            "SubModules.TextDetection.thresh": self._params["text_det_thresh"],
            "SubModules.TextDetection.box_thresh": self._params["text_det_box_thresh"],
            "SubModules.TextDetection.unclip_ratio": self._params["text_det_unclip_ratio"],
            "SubModules.TextDetection.input_shape": self._params["text_det_input_shape"],
            "SubModules.TextRecognition.score_thresh": self._params["text_rec_score_thresh"],
            "SubModules.TextRecognition.return_word_box": self._params["return_word_box"],
            "SubModules.TextRecognition.input_shape": self._params["text_rec_input_shape"]
        }
        return create_config_from_structure(structure)

    def _get_ocr_model_names(self, lang, ppocr_version, use_mobile_model: bool):
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

        model_type = "mobile" if use_mobile_model else "server"

        if ppocr_version == "PP-OCRv5":
            rec_lang, rec_model_name = None, None
            if lang in ("ch", "chinese_cht", "japan"):
                rec_model_name = f"PP-OCRv5_{model_type}_rec"
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
            return f"PP-OCRv5_{model_type}_det", rec_model_name

        elif ppocr_version == "PP-OCRv4":
            if lang == "ch":
                return f"PP-OCRv4_{model_type}_det", f"PP-OCRv4_{model_type}_rec"
            elif lang == "en":
                return f"PP-OCRv4_{model_type}_det", "en_PP-OCRv4_mobile_rec"
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
            return f"PP-OCRv3_{model_type}_det", rec_model_name

    @staticmethod
    def load_pipeline_config(pipeline_name: str = "OCR") -> dict:
        """
        Load the pipeline configuration.
        """
        pipeline_path = (Path(__file__).parent / f"pipeline/{pipeline_name}.yaml").resolve()
        with open(pipeline_path, "r", encoding="utf-8") as yaml_file:
            config = yaml.load(yaml_file, Loader=yaml.FullLoader)
        return config

    def _get_merged_config(self):
        pipeline_config = self.load_pipeline_config()
        overrides = self._get_config_overrides()
        return _merge_dicts(pipeline_config, overrides)

    def predict_iter(self, input_):
        return self.ocr_pipline.predict(input_)

    def predict(self, input_):
        result = list(self.predict_iter(input_))
        return result
