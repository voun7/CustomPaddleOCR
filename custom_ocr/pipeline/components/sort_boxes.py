import numpy as np


class SortQuadBoxes:
    """SortQuadBoxes Operator."""

    entities = "SortQuadBoxes"

    def __init__(self):
        """Initializes the class."""
        pass

    def __call__(self, dt_polys: list[np.ndarray]) -> np.ndarray:
        """
        Sort quad boxes in order from top to bottom, left to right
        args:
            dt_polys(ndarray):detected quad boxes with shape [4, 2]
        return:
            sorted boxes(ndarray) with shape [4, 2]
        """
        dt_boxes = np.array(dt_polys)
        num_boxes = dt_boxes.shape[0]
        sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
        _boxes = list(sorted_boxes)

        for i in range(num_boxes - 1):
            for j in range(i, -1, -1):
                if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and (
                        _boxes[j + 1][0][0] < _boxes[j][0][0]
                ):
                    tmp = _boxes[j]
                    _boxes[j] = _boxes[j + 1]
                    _boxes[j + 1] = tmp
                else:
                    break
        return _boxes
