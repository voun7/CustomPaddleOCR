from pathlib import Path
from typing import Tuple, TypedDict, Final

MODEL_FILE_PREFIX: Final[str] = "inference"


class ModelPaths(TypedDict, total=False):
    paddle: Tuple[Path, Path]
    onnx: Path


def get_model_paths(model_dir: Path) -> ModelPaths:
    model_dir = Path(model_dir)
    model_paths: ModelPaths = {}
    model_file_prefix = MODEL_FILE_PREFIX
    pd_model_path = None
    if (model_dir / f"{model_file_prefix}.json").exists():
        pd_model_path = model_dir / f"{model_file_prefix}.json"
    elif (model_dir / f"{model_file_prefix}.pdmodel").exists():
        pd_model_path = model_dir / f"{model_file_prefix}.pdmodel"
    if pd_model_path and (model_dir / f"{model_file_prefix}.pdiparams").exists():
        model_paths["paddle"] = (pd_model_path, model_dir / f"{model_file_prefix}.pdiparams")
    if (model_dir / f"{model_file_prefix}.onnx").exists():
        model_paths["onnx"] = model_dir / f"{model_file_prefix}.onnx"
    return model_paths
