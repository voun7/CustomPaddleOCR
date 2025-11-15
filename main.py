from datetime import timedelta
from time import perf_counter

import onnxruntime as ort

ort.preload_dlls()

from custom_ocr import TextDetection

start_time = perf_counter()

model_dir = "models"
img_files = r"C:\Users\Victor\OneDrive\Public\test images"

ocr_fn = TextDetection(model_dir)
results = ocr_fn.predict_iter(img_files)
print(list(results))

print(f"Duration: {timedelta(seconds=round(perf_counter() - start_time))}")
