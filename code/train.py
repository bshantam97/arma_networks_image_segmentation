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


# metrics, reference: https://github.com/VainF/DeepLabV3Plus-Pytorch/blob/master/metrics/stream_metrics.py
class _StreamMetrics(object):
    def __init__(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def update(self, gt, pred):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def get_results(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def to_str(self, metrics):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def reset(self):
        """ Overridden by subclasses """
        raise NotImplementedError()      

class StreamSegMetrics(_StreamMetrics):
    """
    Stream Metrics for Semantic Segmentation Task
    """
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist( lt.flatten(), lp.flatten() )
    
    @staticmethod
    def to_str(results):
        string = "\n"
        for k, v in results.items():
            if k!="Class IoU":
                string += "%s: %f\n"%(k, v)
        
        #string+='Class IoU:\n'
        #for k, v in results['Class IoU'].items():
        #    string += "\tclass %d: %f\n"%(k, v)
        return string

    def _fast_hist(self, label_true, label_pred):
        mask = (label_true >= 0) & (label_true < self.n_classes)
        hist = np.bincount(
            self.n_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.n_classes ** 2,
        ).reshape(self.n_classes, self.n_classes)
        return hist

    def get_results(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return {
                "Overall Acc": acc,
                "Mean Acc": acc_cls,
                "FreqW Acc": fwavacc,
                "Mean IoU": mean_iu,
                "Class IoU": cls_iu,
            }
        
    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))

class AverageMeter(object):
    """Computes average values"""
    def __init__(self):
        self.book = dict()

    def reset_all(self):
        self.book.clear()
    
    def reset(self, id):
        item = self.book.get(id, None)
        if item is not None:
            item[0] = 0
            item[1] = 0

    def update(self, id, val):
        record = self.book.get(id, None)
        if record is None:
            self.book[id] = [val, 1]
        else:
            record[0]+=val
            record[1]+=1

    def get_results(self, id):
        record = self.book.get(id, None)
        assert record is not None
        return record[0] / record[1]


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

