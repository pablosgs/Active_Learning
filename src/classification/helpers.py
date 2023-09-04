import numpy as np
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
import torch

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def split_train_val_test(X, y, train_frac: float, val_frac: float, test_frac: float, RANDOM_SEED=1):
    if 1 - (train_frac + val_frac + test_frac) > 0.001:
        raise ValueError("fractions %f, %f, %f do not add up to 1.0" % (train_frac, val_frac, test_frac))
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=(1 - train_frac), random_state=RANDOM_SEED, shuffle=True
    )
    refactored_test_size = test_frac / (val_frac + test_frac)
    X_val, X_test, y_val, y_test = train_test_split(
        X_val, y_val, test_size=refactored_test_size, random_state=RANDOM_SEED, shuffle=True
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

class DistillationLoss:
    def __init__(self, alpha):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.distillation_loss = nn.KLDivLoss()
        self.temperature = 2
        self.alpha = alpha
        self.soft = torch.nn.Softmax(dim=1)

    def __call__(self, student_logits, student_target_loss, teacher_logits):
        #distillation_loss = - self.distillation_loss(student_logits,
                                                   #teacher_logits)
        distillation_loss = self.modified_kl_div(self.smooth(self.soft(teacher_logits), 2, 1), self.smooth(self.soft(student_logits), 2, 1))

        loss = (1 - self.alpha) * student_target_loss + self.alpha * distillation_loss
        return loss
    
    def modified_kl_div(self, old, new):
        return -torch.mean(torch.sum(old * torch.log(new), 1))
    
    def smooth(self, logits, temp, dim):
        log = logits ** (1 / temp)
        return log / torch.sum(log, dim).unsqueeze(1)