import shutil
import subprocess
import sys
from pathlib import Path

from . import logger
from .model_paths import get_model_paths, MODEL_FILE_PREFIX


def _check_input_dir(input_dir, config_filename):
    if input_dir is None:
        sys.exit("Input directory must be specified")
    if not input_dir.exists():
        sys.exit(f"{input_dir} does not exist")
    if not input_dir.is_dir():
        sys.exit(f"{input_dir} is not a directory")
    model_paths = get_model_paths(input_dir)
    if "paddle" not in model_paths:
        sys.exit("PaddlePaddle model does not exist")
    config_path = input_dir / config_filename
    if not config_path.exists():
        sys.exit(f"{config_path} does not exist")


def _check_paddle2onnx():
    if shutil.which("paddle2onnx") is None:
        raise ModuleNotFoundError("Paddle2ONNX is not available. Please install the plugin first.")


def _run_paddle2onnx(input_dir, output_dir, opset_version, onnx_model_filename):
    model_paths = get_model_paths(input_dir)
    logger.info("Paddle2ONNX conversion starting...")
    cmd = [
        "paddle2onnx",
        "--model_dir", str(model_paths["paddle"][0].parent),
        "--model_filename", str(model_paths["paddle"][0].name),
        "--params_filename", str(model_paths["paddle"][1].name),
        "--save_file", str(output_dir / onnx_model_filename),
        "--opset_version", str(opset_version),
    ]
    try:
        logger.debug(f"Running Paddle2ONNX command: {' '.join(cmd)}")
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        sys.exit(f"Paddle2ONNX conversion failed with exit code {e.returncode}")
    logger.info("Paddle2ONNX conversion succeeded...\n")


def _copy_config_file(input_dir, output_dir, config_filename):
    src_path = input_dir / config_filename
    dst_path = output_dir / config_filename
    shutil.copy(src_path, dst_path)
    logger.info(f"Copied {src_path} to {dst_path}")


def _copy_additional_files(input_dir, output_dir, additional_filenames):
    for filename in additional_filenames:
        src_path = input_dir / filename
        if not src_path.exists():
            continue
        dst_path = output_dir / filename
        shutil.copy(src_path, dst_path)
        logger.info(f"Copied {src_path} to {dst_path}")


def paddle_to_onnx(paddle_model_dir, onnx_model_dir, opset_version=7):
    onnx_model_filename = f"{MODEL_FILE_PREFIX}.onnx"
    config_filename = f"{MODEL_FILE_PREFIX}.yml"
    additional_filenames = ["scaler.pkl"]

    if not paddle_model_dir:
        sys.exit("PaddlePaddle model directory must be specified")

    paddle_model_dir = Path(paddle_model_dir)
    if not onnx_model_dir:
        onnx_model_dir = paddle_model_dir
    onnx_model_dir = Path(onnx_model_dir)
    logger.debug(f"Input dir: {paddle_model_dir}")
    logger.debug(f"Output dir: {onnx_model_dir}")
    _check_input_dir(paddle_model_dir, config_filename)
    _check_paddle2onnx()
    _run_paddle2onnx(paddle_model_dir, onnx_model_dir, opset_version, onnx_model_filename)
    if not (onnx_model_dir.exists() and onnx_model_dir.samefile(paddle_model_dir)):
        _copy_config_file(paddle_model_dir, onnx_model_dir, config_filename)
        _copy_additional_files(paddle_model_dir, onnx_model_dir, additional_filenames)
