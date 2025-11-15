from functools import wraps
from pathlib import Path

import yaml

from .static_infer import OnnxInfer
from ..utils import logger
from ..utils.convert import paddle_to_onnx
from ..utils.download import ModelManager
from ..utils.model_paths import get_model_paths, MODEL_FILE_PREFIX


class FuncRegister:
    def __init__(self, register_map):
        assert isinstance(register_map, dict)
        self._register_map = register_map

    def __call__(self, key=None):
        """register the decorated func as key in dict"""

        def decorator(func):
            actual_key = key if key is not None else func.__name__
            self._register_map[actual_key] = func
            logger.debug(f"The func ({func.__name__}) has been registered as key ({actual_key}).")

            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapper

        return decorator


class PredictionWrap:
    """Wraps the prediction data and supports get by index."""

    def __init__(self, data: dict, num: int) -> None:
        """Initializes the PredictionWrap with prediction data.

        Args:
            data (dict): A dictionary where keys are string identifiers and values are lists of predictions.
            num (int): The number of predictions, that is length of values per key in the data dictionary.

        Raises:
            AssertionError: If the length of any list in data does not match num.
        """
        assert isinstance(data, dict), "data must be a dictionary"
        for k in data:
            assert len(data[k]) == num, f"{len(data[k])} != {num} for key {k}!"
        self._data = data
        self._keys = data.keys()

    def get_by_idx(self, idx: int):
        """Get the prediction by specified index.

        Args:
            idx (int): The index to get predictions from.

        Returns:
            Dict[str, Any]: A dictionary with the same keys as the input data, but with the values at the specified index.
        """
        return {key: self._data[key][idx] for key in self._keys}


class BasePredictor:

    def __init__(
            self,
            model_save_dir,
            *,
            use_gpu: str = None,
            batch_size: int = 1,
            model_name: str = None,
            onnx_sess_options=None
    ) -> None:
        """Initializes the BasePredictor.

        Args:
            model_save_dir: The directory where the model
                files are saved.
            use_gpu (Optional[str], optional): The device to run the inference
                engine on. Defaults to None and the device will be chosen automatically.
            batch_size (int, optional): The batch size to predict.
                Defaults to 1.
            model_name (Optional[str], optional): Optional model name.
                Defaults to None.
        """
        model_dir = ModelManager(model_save_dir).get_model(model_name)
        self.use_gpu, self.onnx_sess_options = use_gpu, onnx_sess_options
        model_paths = get_model_paths(model_dir)
        if "onnx" not in model_paths:
            paddle_to_onnx(model_dir, model_dir)
        self.model_dir = Path(model_dir)
        self.config = self.load_config(self.model_dir)
        self._use_local_model = True

        if model_name:
            if self.config:
                if self.config["Global"]["model_name"] != model_name:
                    raise ValueError("`model_name` is not consistent with `config`")
            self._model_name = model_name

        self.batch_sampler = self._build_batch_sampler()
        self.result_class = self._get_result_class()

        # alias predict() to the __call__()
        self.predict = self.__call__
        self.batch_sampler.batch_size = batch_size
        logger.debug(f"{self.__class__.__name__}: {self.model_dir}")

    @property
    def model_name(self) -> str:
        """
        Get the model name.

        Returns:
            str: The model name.
        """
        if self.config:
            return self.config["Global"]["model_name"]
        else:
            if hasattr(self, "_model_name"):
                return self._model_name
            else:
                raise AttributeError(f"{repr(self)} has no attribute 'model_name'.")

    def __call__(self, input_, batch_size=None, **kwargs):
        """
        Predict with the input data.

        Args:
            input_ (Any): The input data to be predicted.
            batch_size (int, optional): The batch size to use. Defaults to None.
            **kwargs (Dict[str, Any]): Additional keyword arguments to set up predictor.

        Returns:
            Iterator[Any]: An iterator yielding the prediction output.
        """
        self.set_predictor(batch_size)
        yield from self.apply(input_, **kwargs)

    def set_predictor(self, batch_size=None) -> None:
        """
        Sets the predictor configuration.

        Args:
            batch_size (Optional[int], optional): The batch size to use. Defaults to None.

        Returns:
            None
        """
        if batch_size:
            self.batch_sampler.batch_size = batch_size

    def create_static_infer(self):
        return OnnxInfer(self.model_dir, self.use_gpu, self.onnx_sess_options)

    def apply(self, input_, **kwargs):
        """
        Do prediction with the input data and yields predictions.

        Args:
            input_ (Any): The input data to be predicted.

        Yields:
            Iterator[Any]: An iterator yielding prediction results.
        """
        batches = self.batch_sampler(input_)
        for batch_data in batches:
            prediction = self.process(batch_data, **kwargs)
            prediction = PredictionWrap(prediction, len(batch_data))
            for idx in range(len(batch_data)):
                yield self.result_class(prediction.get_by_idx(idx))

    def process(self, batch_data: list) -> dict:
        """process the batch data sampled from BatchSampler and return the prediction result.

        Args:
            batch_data (List[Any]): The batch data sampled from BatchSampler.

        Returns:
            dict: The prediction result.
        """
        raise NotImplementedError

    @classmethod
    def load_config(cls, model_dir):
        """Load the configuration from the specified model directory.

        Args:
            model_dir (Path): The where the static model files is stored.

        Returns:
            dict: The loaded configuration dictionary.
        """
        config_file = model_dir / f"{MODEL_FILE_PREFIX}.yml"
        with open(config_file, "r", encoding="utf-8") as yaml_file:
            data = yaml.load(yaml_file, Loader=yaml.FullLoader)
        return data

    def _build_batch_sampler(self):
        """Build batch sampler.

        Returns:
            BaseBatchSampler: batch sampler object.
        """
        raise NotImplementedError

    def _get_result_class(self) -> type:
        """Get the result class.

        Returns:
            type: The result class.
        """
        raise NotImplementedError
