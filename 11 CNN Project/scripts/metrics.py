import tensorflow as tf
import torch

def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct       = (rounded_preds == y).sum() 
    acc           = torch.mean(torch.eq(preds, y).float())
    return acc

def accuracy(prediction, targets):
    rounded_prediction = torch.round(torch.sigmoid(prediction))
    correct = (prediction == targets[0 : 3249]).float()
    return (correct.sum()/len(correct)).item()
