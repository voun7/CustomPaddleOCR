from .processors import NormalizeImage, DetResizeForTest, DBPostProcess
from ..base_predictor import BasePredictor, FuncRegister
from ..batch_sampler import ImageBatchSampler
from ..image_reader import ReadImage
from ..pre_processes import ToBatch, ToCHWImage
from ...results import TextDetResult


class TextDetPredictor(BasePredictor):
    _FUNC_MAP = {}
    register = FuncRegister(_FUNC_MAP)

    def __init__(self, box_thresh: float = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.box_thresh = box_thresh
        self.pre_tfs, self.infer, self.post_op = self._build()

    def _build_batch_sampler(self):
        return ImageBatchSampler()

    def _get_result_class(self):
        return TextDetResult

    def _build(self):
        pre_tfs = {"Read": ReadImage(format_="RGB")}
        for cfg in self.config["PreProcess"]["transform_ops"]:
            tf_key = list(cfg.keys())[0]
            func = self._FUNC_MAP[tf_key]
            args = cfg.get(tf_key, {})
            name, op = func(self, **args) if args else func(self)
            if op:
                pre_tfs[name] = op
        pre_tfs["ToBatch"] = ToBatch()
        infer = self.create_static_infer()
        post_op = self.build_postprocess(**self.config["PostProcess"])
        return pre_tfs, infer, post_op

    def build_postprocess(self, **kwargs):
        if kwargs.get("name") == "DBPostProcess":
            return DBPostProcess(
                thresh=kwargs.get("thresh", 0.3),
                box_thresh=self.box_thresh or kwargs.get("box_thresh", 0.6),
                unclip_ratio=kwargs.get("unclip_ratio", 2.0),
                max_candidates=kwargs.get("max_candidates", 1000),
                use_dilation=kwargs.get("use_dilation", False),
                score_mode=kwargs.get("score_mode", "fast"),
                box_type=kwargs.get("box_type", "quad"),
            )

        else:
            raise Exception()

    def process(self, batch_data):
        batch_raw_imgs = self.pre_tfs["Read"](imgs=batch_data.instances)
        batch_imgs, batch_shapes = self.pre_tfs["Resize"](imgs=batch_raw_imgs)
        batch_imgs = self.pre_tfs["Normalize"](imgs=batch_imgs)
        batch_imgs = self.pre_tfs["ToCHW"](imgs=batch_imgs)
        x = self.pre_tfs["ToBatch"](imgs=batch_imgs)
        batch_preds = self.infer(x)
        polys, scores = self.post_op(batch_preds, batch_shapes)
        return {
            "input_path": batch_data.input_paths,
            "page_index": batch_data.page_indexes,
            "input_img": batch_raw_imgs,
            "dt_polys": polys,
            "dt_scores": scores,
        }

    @register("DecodeImage")
    def build_read_img(self, channel_first, img_mode):
        assert channel_first == False
        return "Read", ReadImage(format_=img_mode)

    @register("DetResizeForTest")
    def build_resize(self, **kwargs):
        limit_side_len = kwargs.get("resize_long", 960)
        limit_type = kwargs.get("limit_type", "max")
        return "Resize", DetResizeForTest(limit_side_len=limit_side_len, limit_type=limit_type, **kwargs)

    @register("NormalizeImage")
    def build_normalize(self, mean, std, scale, order):
        return "Normalize", NormalizeImage(mean=mean, std=std, scale=scale, order=order)

    @register("ToCHWImage")
    def build_to_chw(self):
        return "ToCHW", ToCHWImage()

    @register("DetLabelEncode")
    def foo(self, *args, **kwargs):
        return None, None

    @register("KeepKeys")
    def foo(self, *args, **kwargs):
        return None, None
