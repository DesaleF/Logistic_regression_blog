# Binary classification with Logistic regression

This repository contains a jupyter notebook for a very detailed blog post about binary classification. The post have both theoretical background on how to implement logistics regression for binary classification with example code implemented using numpy. The link for the full blogpost can be found <a href="https://deepcomputervision.blogspot.com/2020/07/building-binary-classifier-from-scratch.html">here </a>


## How to run

### Clone repository
```shell
$ git clone https://github.com/DesaleF/Logistic_regression_blog.git
$ cd Logistic_regression_blog
$ mkdir checkpoints

```

### Install requirements

```shell 
$ conda create --name bic python=3.8
$ conda activate bic
$ pip install -r requirements.txt

```

```shell
$ python datasets/datasets.py
$ python train.py
$ python inference.py --image_path datasets/cat.jpeg
```

### Directory structure

```shell 
.
├── datasets
│   ├── __init__.py
│   └── datasets.py
├── model
│   ├── __init__.py
│   └── model.py
├── notebooks
│   └── Logistic_regression.ipynb
├── __init__.py
├── README.md
├── requirements.txt
├── inference.py
└── train.py

```