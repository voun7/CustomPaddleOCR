# Custom Paddle OCR

![python version](https://img.shields.io/badge/Python-3.12-blue)
![support os](https://img.shields.io/badge/OS-Windows-green.svg)

## Introduction

A program that uses paddleocr models with onnx for OCR inference.

### Install packages

**For Converting paddle models to onnx**

These packages are optional if the model being used has already been converted.

```
pip install paddlepaddle==3.1.1
```

```
pip install paddle2onnx==2.0.2rc3
```

For GPU

```
pip install onnxruntime-gpu[cuda,cudnn]==1.23.2
```

For CPU

```
pip install onnxruntime==1.23.2
```

Other packages

```commandline
pip install -r requirements.txt
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

```
pip install git+https://github.com/voun7/CustomPaddleOCR.git
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