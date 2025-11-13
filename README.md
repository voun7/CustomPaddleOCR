# Custom Paddle OCR

![python version](https://img.shields.io/badge/Python-3.12-blue)
![support os](https://img.shields.io/badge/OS-Windows-green.svg)

## Introduction

A program that uses paddleocr models with onnx for OCR inference.

### Install packages

**For Converting paddle models to onnx**

These packages are optional if the model has already been converted.

```
pip install paddlepaddle==3.1.1
```

```
pip install paddle2onnx==2.0.2rc3
```

For GPU

```
pip install onnxruntime-gpu[cuda,cudnn]
```

For CPU

```
pip install onnxruntime
```

[//]: # (Other packages)

[//]: # ()

[//]: # (```commandline)

[//]: # (pip install -r requirements.txt)

[//]: # (```)

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

Sources: [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR), [PaddleX](https://github.com/PaddlePaddle/PaddleX)