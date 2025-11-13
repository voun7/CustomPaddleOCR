import yaml

from .model_paths import MODEL_FILE_PREFIX


def load_config(model_dir) -> dict:
    """
    Load the configuration from the specified model directory.

    Args:
        model_dir (Path): The where the static model files is stored.

    Returns:
        dict: The loaded configuration dictionary.
    """
    config_file = model_dir / f"{MODEL_FILE_PREFIX}.yml"
    with open(config_file, "r", encoding="utf-8") as yaml_file:
        data = yaml.load(yaml_file, Loader=yaml.FullLoader)
    return data
