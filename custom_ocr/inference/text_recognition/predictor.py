import numpy as np
from bidi.algorithm import get_display

from .processors import CTCLabelDecode, OCRReisizeNormImg, ToBatch
from ..base_predictor import BasePredictor, FuncRegister
from ..batch_sampler import ImageBatchSampler
from ..image_reader import ReadImage
from ...results.fonts import Font
from ...results.text_rec import TextRecResult


class TextRecPredictor(BasePredictor):
    _FUNC_MAP = {}
    register = FuncRegister(_FUNC_MAP)

    def __init__(self, input_shape=None, return_word_box=False, **kwargs):
        super().__init__(**kwargs)
        self.input_shape = input_shape
        self.return_word_box = return_word_box
        self.vis_font = self.get_vis_font()
        self.pre_tfs, self.infer, self.post_op = self._build()

    def _build_batch_sampler(self):
        return ImageBatchSampler()

    def _get_result_class(self):
        return TextRecResult

    def _build(self):
        pre_tfs = {"Read": ReadImage(format_="RGB")}
        for cfg in self.config["PreProcess"]["transform_ops"]:
            tf_key = list(cfg.keys())[0]
            assert tf_key in self._FUNC_MAP
            func = self._FUNC_MAP[tf_key]
            args = cfg.get(tf_key, {})
            name, op = func(self, **args) if args else func(self)
            if op:
                pre_tfs[name] = op
        pre_tfs["ToBatch"] = ToBatch()

        infer = self.create_static_infer()

        post_op = self.build_postprocess(**self.config["PostProcess"])
        return pre_tfs, infer, post_op

    def process(self, batch_data, return_word_box=False):
        batch_raw_imgs = self.pre_tfs["Read"](imgs=batch_data.instances)
        width_list = []
        for img in batch_raw_imgs:
            width_list.append(img.shape[1] / float(img.shape[0]))
        indices = np.argsort(np.array(width_list))
        batch_imgs = self.pre_tfs["ReisizeNorm"](imgs=batch_raw_imgs)
        x = self.pre_tfs["ToBatch"](imgs=batch_imgs)
        batch_preds = self.infer(x=x)
        batch_num = self.batch_sampler.batch_size
        img_num = len(batch_raw_imgs)
        rec_image_shape = next(
            op["RecResizeImg"]["image_shape"]
            for op in self.config["PreProcess"]["transform_ops"]
            if "RecResizeImg" in op
        )
        imgC, imgH, imgW = rec_image_shape[:3]
        max_wh_ratio = imgW / imgH
        end_img_no = min(img_num, batch_num)
        wh_ratio_list = []
        for ino in range(0, end_img_no):
            h, w = batch_raw_imgs[indices[ino]].shape[0:2]
            wh_ratio = w * 1.0 / h
            max_wh_ratio = max(max_wh_ratio, wh_ratio)
            wh_ratio_list.append(wh_ratio)
        texts, scores = self.post_op(
            batch_preds,
            return_word_box=return_word_box or self.return_word_box,
            wh_ratio_list=wh_ratio_list,
            max_wh_ratio=max_wh_ratio,
        )
        if self.model_name in ("arabic_PP-OCRv3_mobile_rec", "arabic_PP-OCRv5_mobile_rec"):
            texts = [get_display(s) for s in texts]
        return {
            "input_path": batch_data.input_paths,
            "page_index": batch_data.page_indexes,
            "input_img": batch_raw_imgs,
            "rec_text": texts,
            "rec_score": scores,
            "vis_font": [self.vis_font] * len(batch_raw_imgs),
        }

    @register("DecodeImage")
    def build_read_img(self, channel_first, img_mode):
        assert channel_first == False
        return "Read", ReadImage(format_=img_mode)

    @register("RecResizeImg")
    def build_resize(self, image_shape, **kwargs):
        return "ReisizeNorm", OCRReisizeNormImg(rec_image_shape=image_shape, input_shape=self.input_shape)

    def build_postprocess(self, **kwargs):
        if kwargs.get("name") == "CTCLabelDecode":
            return CTCLabelDecode(character_list=kwargs.get("character_dict"))
        else:
            raise Exception()

    @register("MultiLabelEncode")
    def foo(self, *args, **kwargs):
        return None, None

    @register("KeepKeys")
    def foo(self, *args, **kwargs):
        return None, None

    def get_vis_font(self):
        if self.model_name.startswith(("PP-OCR", "en_PP-OCR")):
            return Font(font_name="simfang.ttf")

        if self.model_name in (
                "latin_PP-OCRv3_mobile_rec",
                "latin_PP-OCRv5_mobile_rec",
        ):
            return Font(font_name="latin.ttf")

        if self.model_name in (
                "cyrillic_PP-OCRv3_mobile_rec",
                "cyrillic_PP-OCRv5_mobile_rec",
                "eslav_PP-OCRv5_mobile_rec",
        ):
            return Font(font_name="cyrillic.ttf")

        if self.model_name in (
                "korean_PP-OCRv3_mobile_rec",
                "korean_PP-OCRv5_mobile_rec",
        ):
            return Font(font_name="korean.ttf")

        if self.model_name == "th_PP-OCRv5_mobile_rec":
            return Font(font_name="th.ttf")

        if self.model_name == "el_PP-OCRv5_mobile_rec":
            return Font(font_name="el.ttf")

        if self.model_name in (
                "arabic_PP-OCRv3_mobile_rec",
                "arabic_PP-OCRv5_mobile_rec",
        ):
            return Font(font_name="arabic.ttf")

        if self.model_name == "ka_PP-OCRv3_mobile_rec":
            return Font(font_name="kannada.ttf")

        if self.model_name in ("te_PP-OCRv3_mobile_rec", "te_PP-OCRv5_mobile_rec"):
            return Font(font_name="telugu.ttf")

        if self.model_name in ("ta_PP-OCRv3_mobile_rec", "ta_PP-OCRv5_mobile_rec"):
            return Font(font_name="tamil.ttf")

        if self.model_name in (
                "devanagari_PP-OCRv3_mobile_rec",
                "devanagari_PP-OCRv5_mobile_rec",
        ):
            return Font(font_name="devanagari.ttf")
