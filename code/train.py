# header files
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import random
from random import shuffle
from PIL import Image
from collections import namedtuple
import json
from sklearn.metrics import confusion_matrix

from dataset import Cityscapes
from metrics import StreamSegMetrics

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt


# set-up metrics
metrics = StreamSegMetrics(19)
train_loss_list = []
train_accuracy_list = []
train_iou_list = []
val_loss_list = []
val_accuracy_list = []
val_iou_list = []

# training and val loop
for epoch in range(0, 1000):

  # train
  metrics.reset()
  model.train()
  train_loss = 0.0
  for step, (images, labels) in enumerate(train_loader):
    
    # if cuda
    images = images.to(device, dtype=torch.float32)
    labels = labels.to(device, dtype=torch.long)
    labels = labels.squeeze(1)
    
    # get loss
    optimizer.zero_grad()
    outputs = model(images)

    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    train_loss += loss.item()

    # metrics
    preds = outputs.detach().max(dim=1)[1].cpu().numpy()
    targets = labels.cpu().numpy()
    metrics.update(targets, preds)

  # update training_loss, training_accuracy and training_iou 
  train_loss = train_loss/float(len(train_loader))
  train_loss_list.append(train_loss)
  results = metrics.get_results()
  train_accuracy = results["Overall Acc"]
  train_iou = results["Mean IoU"]
  train_accuracy_list.append(train_accuracy)
  train_iou_list.append(train_iou)

  
  # evaluation code
  metrics.reset()
  model.eval()
  val_loss = 0.0
  for step, (images, labels) in enumerate(val_loader):
    with torch.no_grad():

      # if cuda
      images = images.to(device, dtype=torch.float32)
      labels = labels.to(device, dtype=torch.long)
      labels = labels.squeeze(1)

      # get loss
      outputs = model(images)
      loss = criterion(outputs, labels)
      val_loss += loss.item()

      # metrics
      preds = outputs.detach().max(dim=1)[1].cpu().numpy()
      targets = labels.cpu().numpy()
      metrics.update(targets, preds)

  # update val_loss, val_accuracy and val_iou 
  val_loss = val_loss / float(len(val_loader))
  val_loss_list.append(val_loss)
  results = metrics.get_results()
  val_accuracy = results["Overall Acc"]
  val_iou = results["Mean IoU"] 
  val_accuracy_list.append(val_accuracy)
  val_iou_list.append(val_iou)


  print()
  print("Epoch: " + str(epoch))
  print("Training Loss: " + str(train_loss) + "    Validation Loss: " + str(val_loss))
  print("Training Accuracy: " + str(train_accuracy) + "    Validation Accuracy: " + str(val_accuracy))
  print("Training mIoU: " + str(train_iou) + "    Validhation mIoU: " + str(val_iou))
  print()

