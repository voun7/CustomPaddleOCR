# Custom Paddle OCR

![python version](https://img.shields.io/badge/Python-3.12-blue)
![support os](https://img.shields.io/badge/OS-Windows-green.svg)

## Introduction

A program that uses PaddleOCR models with ONNX for OCR inference.

### Install packages

For GPU

```
pip install onnxruntime-gpu[cuda,cudnn]==1.23.2
```

For CPU

```
pip install onnxruntime==1.23.2
```

**For Converting paddle models to onnx**

These packages are optional if the model being used has already been converted.

```
pip install .[full]
```

**For only inference**

```
pip install .
```

### Build Package

```commandline
pip install build
```

```commandline
python -m build
```

## Usage

### Install using pip

Remove `custom_ocr[full]@` to not install optional packages

```
pip install custom_ocr[full]@git+https://github.com/voun7/CustomPaddleOCR.git
```

``` python
from custom_ocr import CustomPaddleOCR

ocr_fn = CustomPaddleOCR(model_save_dir="models")
results = ocr_fn.predict_iter("test image folder")
for res in results:
    print(res)
    res.save_to_img("output")
    res.save_to_json("output")
    
```

Sources: [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR), [PaddleX](https://github.com/PaddlePaddle/PaddleX)