import os
import json
import torch
import numpy as np
from src.classification.helpers import EarlyStopper, DistillationLoss
from tqdm.auto import tqdm

from src.classification.models import SBERTModel
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, TensorDataset
import torchcontrib
from sklearn.metrics import accuracy_score
import torch.optim.swa_utils as swa
from copy import deepcopy
from torch.autograd import Variable
import torch.nn as nn



class EWC(object):
    def __init__(self, model, data):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=0)
        X_lab, X_unlab, X_test, y_lab, y_unlab, y_test, labels = data
        #X_lab, X_test, y_lab, y_test, labels = data
        batch_test = X_test.shape[0]
        batch_size = 32
        self.threshold = 0.5
        self.labels = labels
        torch_weights = torch.from_numpy((len(y_lab)-np.sum(y_lab,axis=0))/np.sum(y_lab,axis=0))
        

        training_data = TensorDataset(
            torch.from_numpy(X_lab.astype(np.float32)), torch.from_numpy(y_lab.astype(np.float32))
        )
        unlabeled_data = TensorDataset(
            torch.from_numpy(X_unlab.astype(np.float32)), torch.from_numpy(y_unlab.astype(np.float32))
        )

        test_data = TensorDataset(
            torch.from_numpy(X_test.astype(np.float32)), torch.from_numpy(y_test.astype(np.float32))
        )

        self.training_loader = DataLoader(training_data, batch_size=batch_size, shuffle = False)
        self.unlabeled_loader = DataLoader(unlabeled_data, batch_size=X_unlab.shape[0], shuffle = False)
        self.validation_loader = DataLoader(unlabeled_data, batch_size=X_unlab.shape[0], shuffle = False)
        self.test_loader = DataLoader(test_data, batch_size=X_test.shape[0], shuffle = False)
        


        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = self._diag_fisher()

        for n, p in self.model.named_parameters():
            self._means[n] = p.data.clone()

    def _diag_fisher(self):
        precision_matrices = {}
        # Set it to zero
        for n, p in self.model.named_parameters():
            params = p.clone().data.zero_()
            precision_matrices[n] = params

        self.model.eval()
        for idx, batch in enumerate(self.training_loader):
            #if not self.mute:
                #print("\t### Batch {} out of {} ###".format(idx + 1, len(self.training_loader)))

            b_inputs = batch[0].to(self.device)
            b_labels = batch[1].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model.forward(b_inputs)
            #label = (outputs > 0.5).astype(int)
            loss = torch.nn.functional.binary_cross_entropy(outputs, b_labels)
            loss.backward()

            for n, p in self.model.named_parameters():
                precision_matrices[n].data += p.grad.data ** 2 / len(self.training_loader)

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def ewc_penalty(self):
        loss = 0
        for n, p in self.model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()
        return loss

def train_ewc(model: nn.Module, optimizer: torch.optim, loss_fn: torch.nn, data_loader: torch.utils.data.DataLoader, ewc: EWC, importance = 0):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_loss = 0
    model.train()

    for idx, batch in enumerate(data_loader):
        #if not self.mute:
            #print("\t### Batch {} out of {} ###".format(idx + 1, len(self.training_loader)))

        b_inputs = batch[0].to(device)
        b_labels = batch[1].to(device)

        optimizer.zero_grad()
        outputs = model.forward(b_inputs)
        loss = loss_fn(outputs, b_labels)
        #for name, param in model.named_parameters():
            #fisher = fisher_matrices[name].to(self.device)
            #opt_param = opt_params[name].to(self.device)
        final_loss = loss + importance * ewc.ewc_penalty()
        final_loss.backward()

        train_loss += final_loss.item()
        #torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        optimizer.step()


    avg_train_loss = train_loss / len(data_loader)
    #print("\t-Average training loss: {0:.2f}".format(avg_train_loss))
    return avg_train_loss