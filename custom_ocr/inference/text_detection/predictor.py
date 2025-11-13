from ...utils.convert import paddle_to_onnx
from ...utils.download import ModelManager
from ...utils.model_paths import get_model_paths


class TextDetection:

    def __init__(self, model_save_dir: str, model_name: str = "PP-OCRv5_server_det") -> None:
        model_dir = ModelManager(model_save_dir).get_model(model_name)
        model_paths = get_model_paths(model_dir)
        if "onnx" not in model_paths:
            paddle_to_onnx(model_dir, model_dir)
