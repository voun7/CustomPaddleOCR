from pathlib import Path

import onnxruntime as ort

from ..utils.model_paths import get_model_paths


class OnnxInfer:
    def __init__(self, model_dir: Path, use_gpu: None | bool, sess_options, onnx_providers):
        self.model_file = get_model_paths(model_dir)["onnx"]
        self.use_gpu, self.sess_options, self.onnx_providers = use_gpu, sess_options, onnx_providers
        self.ort_sess = self._create_sess()

    def __call__(self, x):
        input_name = self.ort_sess.get_inputs()[0].name
        prediction = self.ort_sess.run(None, {input_name: x[0]})
        return prediction

    def _create_sess(self):
        if self.use_gpu is None:
            self.use_gpu = True if ort.get_device() == "GPU" else False

        if self.onnx_providers:
            sess = ort.InferenceSession(self.model_file, sess_options=self.sess_options, providers=self.onnx_providers)
        elif self.use_gpu:
            sess = ort.InferenceSession(
                self.model_file,
                sess_options=self.sess_options,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']  # prefer CUDA Provider over CPU Provider
            )
        else:
            sess = ort.InferenceSession(
                self.model_file,
                sess_options=self.sess_options,
                providers=["CPUExecutionProvider"]
            )
        return sess
