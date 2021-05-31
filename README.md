<p align="center">
    <img alt="logo" src="assets/nadl.gif" />
</p>

<p align="center">
<img alt="Python" src="https://img.shields.io/badge/python%20-%2314354C.svg?&style=for-the-badge&logo=python&logoColor=white"/>
<img alt="NumPy" src="https://img.shields.io/badge/numpy%20-%23013243.svg?&style=for-the-badge&logo=numpy&logoColor=white" />
<img alt="License" src="https://img.shields.io/github/license/siAyush/nadl?style=for-the-badge"/>
</p>


## About

Naive Automatic Differentiation Library (nadl) a small framework that can perform automatic differentiation.
This framework is very simple implementation of how other big framworks do gradient calculation using
Numerical Automatic Differentiation.


## Installation

```
$ git clone https://github.com/siAyush/nadl.git
$ cd nadl
$ python3 setup.py install
```


## Getting Started

The quickest way to start using nadl:
- Clone this repository and install.
- Make a python file and import ```nadl```.
- Write your code.

Example:
```python
from nadl.tensor import Tensor

t1 = Tensor([1, 2, 3], requires_grad=True)
t2 = Tensor([4, 5, 6], requires_grad=True)
t3 = t1 + t2
```


## Testing

I have added a few basic unit tests. The command should be executed in the repository's root 
directory to run the test:
```
$ git clone https://github.com/siAyush/nadl.git
$ cd nadl
$ pytest
```
