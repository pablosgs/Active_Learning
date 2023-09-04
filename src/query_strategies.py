import random
import numpy as np
import math

              
def random_query(preds, batch_size):
    indexes = random.sample(range(len(preds) - 1), batch_size)
    
    return indexes
    
def min_confidence(preds, batch_size):
    confidences = abs(0.5-preds)
    scores = np.min(confidences, axis = 1)
    max_idx = np.argpartition(-scores, batch_size-1, axis=0)[:batch_size]
    return max_idx

def max_score(preds, batch_size):
    class_pred = (preds > 0.5).astype(int)
    scores = preds*(class_pred - 1/2)
    classwise_max = np.max(scores, axis=1)
    max_idx = np.argpartition(-classwise_max, batch_size-1, axis=0)[:batch_size]
    return max_idx

def max_entropy(preds, batch_size):
    scores = -preds*np.log2(preds) - (1-preds)*np.log2(1-preds)
    classwise_max = np.max(scores, axis=1)
    max_idx = np.argpartition(-classwise_max, batch_size-1, axis=0)[:batch_size]
    return max_idx

def mean_entropy(preds, batch_size):
    scores = -preds*np.log2(preds) - (1-preds)*np.log2(1-preds)
    classwise_max = np.mean(scores, axis=1)
    max_idx = np.argpartition(-classwise_max, batch_size-1, axis=0)[:batch_size]
    return max_idx

def label_cardinality(preds, batch_size):
    class_pred = (preds > 0.5).astype(int)
    scores = np.square(np.transpose(class_pred)[0] - np.mean(np.transpose(class_pred)[0])) + np.square(np.transpose(class_pred)[1] - np.mean(np.transpose(class_pred)[1])) + np.square(np.transpose(class_pred)[2] - np.mean(np.transpose(class_pred)[2])) + np.square(np.transpose(class_pred)[3] - np.mean(np.transpose(class_pred)[3])) + np.square(np.transpose(class_pred)[4] - np.mean(np.transpose(class_pred)[4])) + np.square(np.transpose(class_pred)[5] - np.mean(np.transpose(class_pred)[5]))
    max_idx = np.argpartition(-scores, batch_size-1, axis=0)[:batch_size]
    return max_idx

