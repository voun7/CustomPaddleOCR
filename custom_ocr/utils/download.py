import os
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path

import requests

from . import logger


def print_progress(iteration: int, total: int, prefix: str = '', suffix: str = 'Complete', decimals: int = 3,
                   bar_length: int = 25) -> None:
    """
    Call in a loop to create standard out progress bar.
    :param iteration: current iteration
    :param total: total iterations
    :param prefix: a prefix string to be printed in progress bar
    :param suffix: suffix string
    :param decimals: positive number of decimals in percent complete
    :param bar_length: character length of bar
    """
    if not total:  # prevent error if total is zero.
        return

    format_str = "{0:." + str(decimals) + "f}"  # format the % done number string
    percents = format_str.format(100 * (iteration / float(total)))  # calculate the % done
    filled_length = int(round(bar_length * iteration / float(total)))  # calculate the filled bar length
    bar = '#' * filled_length + '-' * (bar_length - filled_length)  # generate the bar string
    print(f"\r{prefix} |{bar}| {percents}% {suffix}", end='', flush=True)  # prints progress on the same line

    if "100.0" in percents:  # prevent next line from joining previous line
        print()


def _download(url, save_path):
    logger.info(f"Connecting to {url} ...")

    with requests.get(url, stream=True, timeout=15) as r:
        r.raise_for_status()

        total_length = r.headers.get("content-length")

        if total_length is None:
            with open(save_path, "wb") as f:
                shutil.copyfileobj(r.raw, f)
        else:
            with open(save_path, "wb") as f:
                dl = 0
                total_length = int(total_length)
                logger.info(f"Downloading {os.path.basename(save_path)}...")
                for data in r.iter_content(chunk_size=4096):
                    dl += len(data)
                    f.write(data)
                    print_progress(dl, total_length)


def _extract_zip_file(file_path, extd_dir):
    """extract zip file"""
    with zipfile.ZipFile(file_path, "r") as f:
        file_list = f.namelist()
        total_num = len(file_list)
        for index, file in enumerate(file_list):
            f.extract(file, extd_dir)
            yield total_num, index


def _extract_tar_file(file_path, extd_dir):
    """
    extract tar file
    """
    try:
        with tarfile.open(file_path, "r:*") as f:
            file_list = f.getnames()
            total_num = len(file_list)
            for index, file in enumerate(file_list):
                try:
                    f.extract(file, extd_dir)
                except KeyError:
                    logger.info(f"File {file} not found in the archive.")
                yield total_num, index
    except Exception as e:
        logger.exception(f"An error occurred: {e}")


def _extract(file_path, extd_dir):
    """extract"""
    logger.info(f"Extracting {os.path.basename(file_path)}")

    if zipfile.is_zipfile(file_path):
        handler = _extract_zip_file
    elif tarfile.is_tarfile(file_path):
        handler = _extract_tar_file
    else:
        raise RuntimeError("Unsupported file format.")

    for total_num, index in handler(file_path, extd_dir):
        print_progress(index + 1, total_num)


def _remove_if_exists(path):
    """remove"""
    if os.path.exists(path):
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)


def download(url, save_path, overwrite=False):
    """download"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if overwrite:
        _remove_if_exists(save_path)
    if not os.path.exists(save_path):
        _download(url, save_path)


def download_and_extract(url, save_dir, dst_name, overwrite=False, no_interm_dir=True):
    """
    download and extract
    NOTE: `url` MUST come from a trusted source, since we do not provide a solution to secure against CVE-2007-4559.
    """
    os.makedirs(save_dir, exist_ok=True)
    dst_path = os.path.join(save_dir, dst_name)
    if overwrite:
        _remove_if_exists(dst_path)

    if not os.path.exists(dst_path):
        with tempfile.TemporaryDirectory() as td:
            arc_file_path = os.path.join(td, url.split("/")[-1])
            extd_dir = os.path.splitext(arc_file_path)[0]
            _download(url, arc_file_path)
            tmp_extd_dir = os.path.join(td, "extract")
            _extract(arc_file_path, tmp_extd_dir)
            if no_interm_dir:
                file_names = os.listdir(tmp_extd_dir)
                if len(file_names) == 1:
                    file_name = file_names[0]
                else:
                    file_name = dst_name
                sp = os.path.join(tmp_extd_dir, file_name)
                if not os.path.exists(sp):
                    raise FileNotFoundError
                dp = os.path.join(save_dir, file_name)
                if os.path.isdir(sp):
                    shutil.copytree(sp, dp, symlinks=True)
                else:
                    shutil.copyfile(sp, dp)
                extd_file = dp
            else:
                shutil.copytree(tmp_extd_dir, extd_dir)
                extd_file = extd_dir

            if not os.path.exists(dst_path) or not os.path.samefile(extd_file, dst_path):
                shutil.move(extd_file, dst_path)


class ModelManager:
    _healthcheck_timeout = 1
    version = "paddle3.0.0"
    base_url = "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model"

    def __init__(self, save_dir: str) -> None:
        self._save_dir = save_dir

    def get_model(self, model_name: str) -> Path:
        model_dir = Path(self._save_dir) / model_name
        if not model_dir.exists():
            logger.info(f"Using model '{model_name}'. The models files will be downloaded and saved."
                        f"\nDir: {model_dir.absolute()}.")
            self._download(model_name, model_dir)
            logger.info(f"'{model_name}' model files has been download from model source!")
        return model_dir

    def _download(self, model_name: str, save_dir: Path) -> None:
        fn = f"{model_name}_infer.tar"
        url = f"{self.base_url}/{self.version}/{fn}"
        download_and_extract(url, save_dir.parent, model_name, overwrite=False)
