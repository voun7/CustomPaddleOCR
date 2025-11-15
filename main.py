import pprint

import onnxruntime as ort

from custom_ocr import TextDetection

ort.preload_dlls()

model_dir = "models"
img_files = r"C:\Users\Victor\OneDrive\Public\test images"

ocr_fn = TextDetection(model_dir)
results = ocr_fn.predict_iter(img_files)
pprint.pprint(list(results))
