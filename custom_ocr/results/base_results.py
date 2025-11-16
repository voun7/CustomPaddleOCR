import copy
import inspect
import json
import mimetypes
import random
import time
from abc import abstractmethod
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ..utils import logger


class StrMixin:
    """Mixin class for adding string conversion capabilities."""

    @property
    def str(self):
        """Property to get the string representation of the result.

        Returns:
            Dict[str, str]: The string representation of the result.
        """

        return self._to_str()

    def _to_str(
            self,
    ):
        """Convert the given result data to a string representation.

        Args:
            json_format (bool): If True, return a JSON formatted string. Default is False.
            indent (int): Number of spaces to indent for JSON formatting. Default is 4.
            ensure_ascii (bool): If True, ensure all characters are ASCII. Default is False.

        Returns:
            Dict[str, str]: The string representation of the result.
        """
        return {"res": self}

    def print(self) -> None:
        """Print the string representation of the result."""
        logger.info(self._to_str())


def _format_data(obj):
    """Helper function to format data into a JSON-serializable format.

    Args:
        obj: The object to be formatted.

    Returns:
        Any: The formatted object.
    """
    if isinstance(obj, np.float32):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return [_format_data(item) for item in obj.tolist()]
    elif isinstance(obj, Path):
        return obj.as_posix()
    elif isinstance(obj, dict):
        return dict({k: _format_data(v) for k, v in obj.items()})
    elif isinstance(obj, (list, tuple)):
        return [_format_data(i) for i in obj]
    else:
        return obj


class JsonMixin:
    """Mixin class for adding JSON serialization capabilities."""

    def __init__(self) -> None:
        self._json_writer = JsonWriter()
        self._save_funcs.append(self.save_to_json)

    def _to_json(self):
        """Convert the object to a JSON-serializable format.

        Returns:
            Dict[str, Dict[str, Any]]: A dictionary representation of the object that is JSON-serializable.
        """

        return {"res": _format_data(copy.deepcopy(self))}

    @property
    def json(self):
        """Property to get the JSON representation of the result.

        Returns:
            Dict[str, Dict[str, Any]]: The dict type JSON representation of the result.
        """

        return self._to_json()

    def save_to_json(
            self,
            save_path: str,
            indent: int = 4,
            ensure_ascii: bool = False,
            *args,
            **kwargs,
    ) -> None:
        """Save the JSON representation of the object to a file.

        Args:
            save_path (str): The path to save the JSON file. If the save path does not end with '.json', it appends the base name and suffix of the input path.
            indent (int): The number of spaces to indent for pretty printing. Default is 4.
            ensure_ascii (bool): If False, non-ASCII characters will be included in the output. Default is False.
            *args: Additional positional arguments to pass to the underlying writer.
            **kwargs: Additional keyword arguments to pass to the underlying writer.
        """

        def _is_json_file(file_path):
            mime_type, _ = mimetypes.guess_type(file_path)
            return mime_type is not None and mime_type == "application/json"

        json_data = self._to_json()
        if not _is_json_file(save_path):
            fn = Path(self._get_input_fn())
            stem = fn.stem
            base_save_path = Path(save_path)
            for key in json_data:
                save_path = base_save_path / f"{stem}_{key}.json"
                self._json_writer.write(
                    save_path.as_posix(),
                    json_data[key],
                    indent=indent,
                    ensure_ascii=ensure_ascii,
                    *args,
                    **kwargs,
                )
        else:
            if len(json_data) > 1:
                logger.warning(
                    f"The result has multiple json files need to be saved. But the `save_path` has been specified as `{save_path}`!"
                )
            self._json_writer.write(
                save_path,
                json_data[list(json_data.keys())[0]],
                indent=indent,
                ensure_ascii=ensure_ascii,
                *args,
                **kwargs,
            )

    def _to_str(
            self,
            json_format: bool = False,
            indent: int = 4,
            ensure_ascii: bool = False,
    ):
        """Convert the given result data to a string representation.
        Args:
            data (dict): The data would be converted to str.
            json_format (bool): If True, return a JSON formatted string. Default is False.
            indent (int): Number of spaces to indent for JSON formatting. Default is 4.
            ensure_ascii (bool): If True, ensure all characters are ASCII. Default is False.
        Returns:
            Dict[str, str]: The string representation of the result.
        """
        if json_format:
            return json.dumps(
                _format_data({"res": self}), indent=indent, ensure_ascii=ensure_ascii
            )
        else:
            return {"res": self}

    def print(
            self, json_format: bool = False, indent: int = 4, ensure_ascii: bool = False
    ) -> None:
        """Print the string representation of the result.

        Args:
            json_format (bool): If True, print a JSON formatted string. Default is False.
            indent (int): Number of spaces to indent for JSON formatting. Default is 4.
            ensure_ascii (bool): If True, ensure all characters are ASCII. Default is False.
        """
        str_ = self._to_str(
            json_format=json_format, indent=indent, ensure_ascii=ensure_ascii
        )
        logger.info(str_)


class ImgMixin:
    """Mixin class for adding image handling capabilities."""

    def __init__(self, backend: str = "pillow", *args, **kwargs) -> None:
        """Initializes ImgMixin.

        Args:
            backend (str): The backend to use for image processing. Defaults to "pillow".
            *args: Additional positional arguments to pass to the ImageWriter.
            **kwargs: Additional keyword arguments to pass to the ImageWriter.
        """
        self._img_writer = ImageWriter(backend=backend, *args, **kwargs)
        self._save_funcs.append(self.save_to_img)

    @abstractmethod
    def _to_img(self):
        """Abstract method to convert the result to an image.

        Returns:
            Dict[str, Image.Image]: The image representation result.
        """
        raise NotImplementedError

    @property
    def img(self):
        """Property to get the image representation of the result.

        Returns:
            Dict[str, Image.Image]: The image representation of the result.
        """
        return self._to_img()

    def save_to_img(self, save_path: str, *args, **kwargs) -> None:
        """Saves the image representation of the result to the specified path.

        Args:
            save_path (str): The path to save the image. If the save path does not end with .jpg or .png, it appends the input path's stem and suffix to the save path.
            *args: Additional positional arguments that will be passed to the image writer.
            **kwargs: Additional keyword arguments that will be passed to the image writer.
        """

        def _is_image_file(file_path):
            mime_type, _ = mimetypes.guess_type(file_path)
            return mime_type is not None and mime_type.startswith("image/")

        img = self._to_img()
        if not _is_image_file(save_path):
            fn = Path(self._get_input_fn())
            suffix = fn.suffix if _is_image_file(fn) else ".png"
            stem = fn.stem
            base_save_path = Path(save_path)
            for key in img:
                save_path = base_save_path / f"{stem}_{key}{suffix}"
                self._img_writer.write(save_path.as_posix(), img[key], *args, **kwargs)
        else:
            if len(img) > 1:
                logger.warning(
                    f"The result has multiple img files need to be saved. But the `save_path` has been specified as `{save_path}`!"
                )
            self._img_writer.write(save_path, img[list(img.keys())[0]], *args, **kwargs)


class WriterType(Enum):
    """WriterType"""

    IMAGE = 1
    JSON = 4


class _BaseJsonWriterBackend:
    def __init__(self, indent=4, ensure_ascii=False):
        super().__init__()
        self.indent = indent
        self.ensure_ascii = ensure_ascii

    def write_obj(self, out_path, obj, **bk_args):
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        return self._write_obj(out_path, obj, **bk_args)

    def _write_obj(self, out_path, obj):
        raise NotImplementedError


class JsonWriterBackend(_BaseJsonWriterBackend):
    def _write_obj(self, out_path, obj, **bk_args):
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(obj, f, **bk_args)


class _BaseWriter:
    """_BaseWriter"""

    def __init__(self, backend, **bk_args):
        super().__init__()
        if len(bk_args) == 0:
            bk_args = self.get_default_backend_args()
        self.bk_type = backend
        self.bk_args = bk_args
        self._backend = self.get_backend()

    def write(self, out_path, obj):
        """write"""
        raise NotImplementedError

    def get_backend(self, bk_args=None):
        """get backend"""
        if bk_args is None:
            bk_args = self.bk_args
        return self._init_backend(self.bk_type, bk_args)

    def set_backend(self, backend, **bk_args):
        self.bk_type = backend
        self.bk_args = bk_args
        self._backend = self.get_backend()

    def _init_backend(self, bk_type, bk_args):
        """init backend"""
        raise NotImplementedError

    def get_type(self):
        """get type"""
        raise NotImplementedError

    def get_default_backend_args(self):
        """get default backend arguments"""
        return {}


class JsonWriter(_BaseWriter):
    def __init__(self, backend="json", **bk_args):
        super().__init__(backend=backend, **bk_args)

    def write(self, out_path, obj, **bk_args):
        return self._backend.write_obj(str(out_path), obj, **bk_args)

    def _init_backend(self, bk_type, bk_args):
        if bk_type == "json":
            return JsonWriterBackend(**bk_args)
        else:
            raise ValueError("Unsupported backend type")

    def get_type(self):
        """get type"""
        return WriterType.JSON


class BaseResult(dict, JsonMixin, StrMixin):
    """Base class for result objects that can save themselves.

    This class inherits from dict and provides properties and methods for handling result.
    """

    def __init__(self, data: dict) -> None:
        """Initializes the BaseResult with the given data.

        Args:
            data (dict): The initial data.
        """
        super().__init__(data)
        self._save_funcs = []
        StrMixin.__init__(self)
        JsonMixin.__init__(self)
        np.set_printoptions(threshold=1, edgeitems=1)
        self._rand_fn = None

    def save_all(self, save_path: str) -> None:
        """Calls all registered save methods with the given save path.

        Args:
            save_path (str): The path to save the result to.
        """
        for func in self._save_funcs:
            signature = inspect.signature(func)
            if "save_path" in signature.parameters:
                func(save_path=save_path)
            else:
                func()

    def _get_input_fn(self):
        if self.get("input_path", None) is None:
            if self._rand_fn:
                return self._rand_fn

            timestamp = int(time.time())
            random_number = random.randint(1000, 9999)
            fp = f"{timestamp}_{random_number}"
            logger.warning(
                f"There is not input file name as reference for name of saved result file. So the saved result file would be named with timestamp and random number: `{fp}`."
            )
            self._rand_fn = Path(fp).name
            return self._rand_fn
        fp = self["input_path"]
        return Path(fp).name


class _BaseWriterBackend:
    """_BaseWriterBackend"""

    def write_obj(self, out_path, obj, **bk_args):
        """write object"""
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        return self._write_obj(out_path, obj, **bk_args)

    def _write_obj(self, out_path, obj, **bk_args):
        """write object"""
        raise NotImplementedError


class _ImageWriterBackend(_BaseWriterBackend):
    """_ImageWriterBackend"""


class OpenCVImageWriterBackend(_ImageWriterBackend):
    """OpenCVImageWriterBackend"""

    def _write_obj(self, out_path, obj):
        """write image object by OpenCV"""
        if isinstance(obj, Image.Image):
            # Assuming the channel order is RGB.
            arr = np.asarray(obj)[:, :, ::-1]
        elif isinstance(obj, np.ndarray):
            arr = obj
        else:
            raise TypeError("Unsupported object type")
        return cv2.imwrite(out_path, arr)


class PILImageWriterBackend(_ImageWriterBackend):
    """PILImageWriterBackend"""

    def __init__(self, format_=None):
        super().__init__()
        self.format = format_

    def _write_obj(self, out_path, obj):
        """write image object by PIL"""
        if isinstance(obj, Image.Image):
            img = obj
        elif isinstance(obj, np.ndarray):
            img = Image.fromarray(obj)
        else:
            raise TypeError("Unsupported object type")
        if len(img.getbands()) == 4:
            self.format = "PNG"
        return img.save(out_path, format=self.format)


class ImageWriter(_BaseWriter):
    """ImageWriter"""

    def __init__(self, backend="opencv", **bk_args):
        super().__init__(backend=backend, **bk_args)

    def write(self, out_path, obj):
        """write"""
        return self._backend.write_obj(str(out_path), obj)

    def _init_backend(self, bk_type, bk_args):
        """init backend"""
        if bk_type == "opencv":
            return OpenCVImageWriterBackend(**bk_args)
        elif bk_type == "pil" or bk_type == "pillow":
            return PILImageWriterBackend(**bk_args)
        else:
            raise ValueError("Unsupported backend type")

    def get_type(self):
        """get type"""
        return WriterType.IMAGE


class BaseCVResult(BaseResult, ImgMixin):
    """Base class for computer vision results."""

    def __init__(self, data: dict) -> None:
        """
        Initialize the BaseCVResult.

        Args:
            data (dict): The initial data.
        """
        super().__init__(data)
        ImgMixin.__init__(self, "pillow")

    def _get_input_fn(self):
        fn = super()._get_input_fn()
        if (page_idx := self.get("page_index", None)) is not None:
            fp = Path(fn)
            stem, suffix = fp.stem, fp.suffix
            fn = f"{stem}_{page_idx}{suffix}"
        return fn
