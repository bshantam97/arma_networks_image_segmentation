# ARMA Networks for Image Segmentation

[![Packagist](https://img.shields.io/packagist/l/doctrine/orm.svg)](LICENSE.md)
---


### Author
Arpit Aggarwal Shantam Bajpai


### Introduction to the Project 
In this research project, we aim to integrate the state of the art semantic segmentation architectures like Deeplab-v3, DeepLab v3+  and RefineNet with Auto Regressive moving average networks and test its performance on the cityscapes dataset.
Auto regressive moving average networks provide a route through which we can expand the effective receptive fields of a Convolution Neural Network which can prove to be benfitial for the task of dense prediction as more information can be encoded in the deeper layer which can lead to accurate segmentations. 

### Data
The dataset used was Cityscapes dataset for the task of image segmentation(number of classes=19). The dataset can be downloaded from here: https://www.cityscapes-dataset.com/downloads/ 


### Software Required
To run the jupyter notebooks, use Python 3. Standard libraries like Numpy and PyTorch are used.


### Credits
The following links were helpful for this project:
1. https://github.com/umd-huang-lab/ARMA-Networks
2. https://github.com/meetshah1995/pytorch-semseg/
3. https://github.com/bodokaiser/piwise/
