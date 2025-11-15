import os
from pathlib import Path

import numpy as np

from ..utils import logger


class ImgBatch:
    def __init__(self):
        self.instances = []
        self.input_paths = []
        self.page_indexes = []

    def append(self, instance, input_path, page_index):
        self.instances.append(instance)
        self.input_paths.append(input_path)
        self.page_indexes.append(page_index)

    def reset(self):
        self.instances = []
        self.input_paths = []
        self.page_indexes = []

    def __len__(self):
        return len(self.instances)


class ImageBatchSampler:
    IMG_SUFFIX = ["jpg", "png", "jpeg", "bmp"]

    def __init__(self, batch_size: int = 1):
        self.batch_size = batch_size

    def __call__(self, input_):
        """
        Sample batch data with the specified input.

        If input is None and benchmarking is enabled, it will yield batches
        of random data for the specified number of iterations.
        Otherwise, it will yield from the apply() function.

        Args:
            input_ (Any): The input data to sample.

        Yields:
            Iterator[List[Any]]: An iterator yielding the batch data.
        """
        yield from self.sample(input_)

    def _get_files_list(self, fp):
        if fp is None or not os.path.exists(fp):
            raise Exception(f"Not found any files in path: {fp}")
        if os.path.isfile(fp):
            return [fp]

        file_list = []
        if os.path.isdir(fp):
            for root, dirs, files in os.walk(fp):
                for single_file in files:
                    if single_file.split(".")[-1].lower() in self.IMG_SUFFIX:
                        file_list.append(os.path.join(root, single_file))
        if len(file_list) == 0:
            raise Exception("Not found any file in {}".format(fp))
        file_list = sorted(file_list)
        return file_list

    def sample(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]

        batch = ImgBatch()
        for input_ in inputs:
            if isinstance(input_, np.ndarray):
                batch.append(input_, None, None)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = ImgBatch()
            elif isinstance(input_, str):
                suffix = input_.split(".")[-1].lower()
                if suffix in self.IMG_SUFFIX:
                    file_path = input_
                    batch.append(file_path, file_path, None)
                    if len(batch) == self.batch_size:
                        yield batch
                        batch = ImgBatch()
                elif Path(input_).is_dir():
                    file_list = self._get_files_list(input_)
                    yield from self.sample(file_list)
                else:
                    logger.error(f"Not supported input file type! Only image files ending with suffix "
                                 f"`{', '.join(self.IMG_SUFFIX)}` are supported! But recevied `{input_}`.")
                    yield batch
            else:
                logger.warning(f"Not supported input data type! Only `numpy.ndarray` and `str` are supported! "
                               f"So has been ignored: {input_}.")
        if len(batch) > 0:
            yield batch
