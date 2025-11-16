from pathlib import Path

from ..utils.download import download


class Font:
    def __init__(self, font_name=None, local_path=None):
        self._local_path = local_path
        if not local_path:
            assert font_name is not None
            self._font_name = font_name

    @property
    def path(self):
        # HACK: download font file when needed only
        if not self._local_path:
            self._get_offical_font()
        return self._local_path

    def _get_offical_font(self):
        """
        Download the official font file.
        """
        font_path = (Path("fonts") / self._font_name).resolve().as_posix()
        if not Path(font_path).is_file():
            download(
                url=f"https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/fonts/{self._font_name}",
                save_path=font_path,
            )
        self._local_path = font_path
