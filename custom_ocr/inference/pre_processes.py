import numpy as np


class ToCHWImage:
    """Reorder the dimensions of the image from HWC to CHW."""

    def __call__(self, imgs):
        """apply"""
        return [img.transpose((2, 0, 1)) for img in imgs]


class ToBatch:
    def __call__(self, imgs):
        return [np.stack(imgs, axis=0).astype(dtype=np.float32, copy=False)]
