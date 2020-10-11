# ARMA Networks for Image Segmentation

[![Packagist](https://img.shields.io/packagist/l/doctrine/orm.svg)](LICENSE.md)
---


### Author
Arpit Aggarwal Shantam Bajpai


### Introduction to the Project 
In this project, different CNN Architectures like ARMA Deeplab-v3, Deeplab-v3, SegNet were used for the task of image segmentation. The input to the CNN networks was a (224x224x3) image and the number of classes were equal to 19. The CNN architectures were implemented in PyTorch and the loss function was Cross Entropy(CE) Loss. The hyperparameters to be tuned were: Number of epochs(e), Learning Rate(lr), weight decay(wd) and batch size(bs).


### Data
The dataset used was Cityscapes dataset for the task of image segmentation(number of classes=19). The dataset can be downloaded from here: https://www.cityscapes-dataset.com/downloads/ 


### Software Required
To run the jupyter notebooks, use Python 3. Standard libraries like Numpy and PyTorch are used.


### Credits
The following links were helpful for this project:
1. https://github.com/umd-huang-lab/ARMA-Networks
2. https://github.com/meetshah1995/pytorch-semseg/
3. https://github.com/bodokaiser/piwise/
