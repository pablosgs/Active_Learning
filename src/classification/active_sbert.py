import os
import json
import torch
import numpy as np
from src.classification.helpers import EarlyStopper, DistillationLoss
from tqdm.auto import tqdm
from torch.nn import functional as F
from src.classification.models import SBERTModel
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, TensorDataset
import torchcontrib
from sklearn.metrics import accuracy_score
import torch.optim.swa_utils as swa
from copy import deepcopy
from torch.autograd import Variable
from src.classification.EWC import EWC, train_ewc
from src.alc_utils import select_probabilities, compute_lambda, _sample_easy, _sample_hard, _update_stat, update_mapping, new_score_leitner, build_training_mask_leitner, compute_new_queues




class ClassificationManagerSBERT:
    def __init__(self, data, mute: bool) -> list:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        
        

        # Setup datasets
        X_lab, X_unlab, X_test, y_lab, y_unlab, y_test, labels = data
        self.train_X = X_lab
        self.train_y = y_lab
        self.train_size = len(X_lab)

        batch_test = X_test.shape[0]
        self.mute = mute
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
        

        # Setup model
        input_layer_size = 384
        hidden_layer_size = 100
        output_layer_size = y_test.shape[1]
        dropout = 0.1
        c_weight = 100  # ​>1 increases the recall, ​<1 increases the precision
        lr = 0.0005
        wd = 0
        self.model = SBERTModel(input_layer_size, hidden_layer_size, output_layer_size, self.device, dropout)
        c_weights = torch.full([len(labels)], c_weight)
        #self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch_weights, reduction="mean", weight=c_weights)  
        self.loss_fn = torch.nn.BCELoss(reduction="mean", weight=c_weights) 
        self.loss_per_sample = torch.nn.BCELoss(reduce=False, weight=c_weights) 
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        self.swa_model = torch.optim.swa_utils.AveragedModel(self.model)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        self.swa_start = 210
        self.swa_scheduler = torch.optim.swa_utils.SWALR(self.optimizer, swa_lr=1e-3)

        self.ewc = EWC(model = self.model, data = data)



    def fit(self, teacher = None, n_epochs = 10, dis = 0, ewc = 0, derpp = 0, datamap = 0, curriculum = 0, leitner = 0, buffer = {}):
        training_stats = []
        early_stopper = EarlyStopper()
        confidence = {}
        #min_val_loss = np.inf



        for epoch in range(n_epochs):
            #print("\n======== Epoch {:} / {:} ========".format(epoch + 1, n_epochs))
            
            #print("######## Epoch {}: Training Start ########".format(epoch + 1))
            if dis == 1:
                avg_train_loss = self.train_distillation(teacher)
            elif ewc == 1:
                avg_train_loss = train_ewc(model = self.model, optimizer = self.optimizer, loss_fn = self.loss_fn, data_loader = self.training_loader, ewc = self.ewc, importance = 1)
            elif derpp == 1:
                avg_train_loss = self.train_derpp(teacher, buffer, alpha = 2000, beta = 1000)
            elif datamap == 1:
                avg_train_loss, epoch_conf = self.train_datamap()
                confidence[str(epoch)] = epoch_conf
            
            else:
                avg_train_loss = self.train_epoch()
            
            
            if curriculum == 1:
                if epoch == 0:
                    scores = np.zeros(self.train_size)
                new_scores, losses, mapping = self.curriculum_spotter(scores, epoch + 1)
                self.new_train_X = self.train_X[np.argwhere(mapping).flatten()]
                self.new_train_y = self.train_y[np.argwhere(mapping).flatten()]
                training_data = TensorDataset(torch.from_numpy(self.new_train_X.astype(np.float32)), torch.from_numpy(self.new_train_y.astype(np.float32)))
                self.training_loader = DataLoader(training_data, batch_size=32, shuffle = False)
                if epoch == n_epochs - 1:
                    scores += (new_scores == 0) * losses


            if leitner == 1:
                if epoch == 0:
                    queues = [[] for _ in range(10)]
                    queues[0] = list(range(len(self.train_X)))
                    scores = np.zeros(self.train_size)
                    training_mask = np.ones(len(self.train_X), dtype=bool)
                new_scores, losses, queues = self.leitner_spotter(scores, epoch + 1, queues, training_mask)

                training_mask = build_training_mask_leitner(training_mask, int(epoch), queues)
                self.new_train_X = self.train_X[np.argwhere(training_mask).flatten()]
                self.new_train_y = self.train_y[np.argwhere(training_mask).flatten()]
                training_data = TensorDataset(torch.from_numpy(self.new_train_X.astype(np.float32)), torch.from_numpy(self.new_train_y.astype(np.float32)))
                self.training_loader = DataLoader(training_data, batch_size=32, shuffle = False)
                if epoch == n_epochs - 1:
                    scores += (new_scores == 0) * losses


            avg_val_loss = self.validate_epoch()

            training_stats.append({"epoch": epoch + 1, "Training Loss": avg_train_loss, 'Valid. Loss': avg_val_loss})

            #if early_stopper.early_stop(avg_val_loss):
            #    break
        #torch.optim.swa_utils.update_bn(self.training_loader, self.swa_model)
        #if avg_val_loss <= min_val_loss:
            #self.save_model(self.model_save_path)
        
        if datamap == 1:
            return training_stats, confidence
        elif curriculum == 1:
            return training_stats, scores, mapping
        elif leitner == 1:
            return training_stats, scores, training_mask
        else:
            #return avg_train_loss
            return training_stats

    def predict_unlab(self):
        y_pred = np.empty(shape=[0, len(self.labels)])

        self.model.eval()
        for batch in self.unlabeled_loader:
            b_inputs = batch[0].to(self.device)

            with torch.no_grad():
                b_y_pred = self.model.forward(b_inputs)
            y_pred = np.append(y_pred, b_y_pred, axis=0)
        return y_pred
    
    def predict_lab(self):
        y_pred = np.empty(shape=[0, len(self.labels)])

        self.model.eval()
        for batch in self.training_loader:
            b_inputs = batch[0].to(self.device)

            with torch.no_grad():
                b_y_pred = self.model.forward(b_inputs)
            y_pred = np.append(y_pred, b_y_pred, axis=0)
        return y_pred
    
    def produce_embeddings(self):
        y_pred = np.empty(shape=[0, 100])

        self.model.eval()
        for batch in self.unlabeled_loader:
            b_inputs = batch[0].to(self.device)

            with torch.no_grad():
                b_y_pred = self.model.embeddings(b_inputs)
            y_pred = np.append(y_pred, b_y_pred, axis=0)
        return y_pred

    def train_epoch(self):

        train_loss = 0
        self.model.train()
        for idx, batch in enumerate(self.training_loader):
            #if not self.mute:
                #print("\t### Batch {} out of {} ###".format(idx + 1, len(self.training_loader)))

            b_inputs = batch[0].to(self.device)
            b_labels = batch[1].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model.forward(b_inputs)
            loss = self.loss_fn(outputs, b_labels)
            loss.backward()

            train_loss += loss.item()
            #torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

        avg_train_loss = train_loss / len(self.training_loader)
        #print("\t-Average training loss: {0:.2f}".format(avg_train_loss))
        return avg_train_loss
    

    def evaluate_model(self):
        
        y_pred = np.empty(shape=[0, len(self.labels)])
        y_true = np.empty(shape=[0, len(self.labels)])

        self.model.eval()
        for batch in self.test_loader:
            b_inputs = batch[0].to(self.device)
            b_labels = batch[1].to(self.device)

            with torch.no_grad():
                b_y_pred = self.model.forward(b_inputs)
                b_y_pred = (b_y_pred.cpu().numpy() > self.threshold).astype(int)
            y_pred = np.append(y_pred, b_y_pred, axis=0)
            y_true = np.append(y_true, b_labels, axis=0)

        accuracy = np.sum(np.sum(y_pred == y_true))/(y_pred.shape[0]*y_pred.shape[1])
        results = classification_report(y_true, y_pred, output_dict=True, target_names=self.labels, zero_division=False)
        return results, accuracy

    def save_model(self, model_save_path: str) -> None:
        torch.save(self.model.state_dict(), model_save_path)

    def validate_epoch(self):
        self.model.eval()
        val_loss = 0
        for idx, batch in enumerate(self.validation_loader):
            if not self.mute:
                print("\t### Batch {} out of {} ###".format(idx + 1, len(self.validation_loader)))
            b_inputs = batch[0].to(self.device)
            b_labels = batch[1].to(self.device)

            with torch.no_grad():
                outputs = self.model.forward(b_inputs)
                #print(torch.mean(torch.mean(outputs)))
                loss = self.loss_fn(outputs, b_labels)

            val_loss += loss.item()

        avg_val_loss = val_loss / len(self.validation_loader)
        #print("\t-Average validation loss: {0:.2f}".format(avg_val_loss))
        return avg_val_loss
    
    def train_distillation(self, teacher):
        criterion = DistillationLoss(0.8)
        train_loss = 0
        self.model.train()
        for idx, batch in enumerate(self.training_loader):
            #if not self.mute:
                #print("\t### Batch {} out of {} ###".format(idx + 1, len(self.training_loader)))

            b_inputs = batch[0].to(self.device)
            b_labels = batch[1].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model.forward(b_inputs)
            teacher_logits = teacher.model.forward(b_inputs)
            loss = self.loss_fn(outputs, b_labels)
            final_loss = criterion(outputs, loss, teacher_logits)
            final_loss.backward()

            train_loss += loss.item()
            #torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

        avg_train_loss = train_loss / len(self.training_loader)
        #print("\t-Average training loss: {0:.2f}".format(avg_train_loss))
        return avg_train_loss
    
    def train_derpp(self, teacher, buffer, alpha = 0, beta = 0):

        buffer_data_logits = TensorDataset(
            torch.from_numpy(buffer['X'][:int(len(buffer['X'])*0.5)].astype(np.float32)), torch.from_numpy(buffer['logits'][:int(len(buffer['X'])*0.5)].astype(np.float32))
        )
        buffer_data_labels = TensorDataset(
            torch.from_numpy(buffer['X'][int(len(buffer['X'])*0.5):].astype(np.float32)), torch.from_numpy(buffer['labels'][int(len(buffer['X'])*0.5):].astype(np.float32))
        )
        #self.buffer_logits = DataLoader(buffer_data_logits, batch_size=buffer_data_logits.shape[0], shuffle = False)
        #self.buffer_labels = DataLoader(buffer_data_labels, batch_size=buffer_data_labels.shape[0], shuffle = False)
        criterion = DistillationLoss(0.2)
        train_loss = 0
        self.model.train()
        n_batches = len(self.training_loader)

        for idx, batch in enumerate(self.training_loader):
            #if not self.mute:
                #print("\t### Batch {} out of {} ###".format(idx + 1, len(self.training_loader)))

            b_inputs = batch[0].to(self.device)
            b_labels = batch[1].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model.forward(b_inputs)
            teacher_logits = teacher.model.forward(b_inputs)
            loss = self.loss_fn(outputs, b_labels)


            #loss_logits = self.args.alpha * F.mse_loss(buf_outputs, buf_logits)
            buffer_X, buffer_logits = buffer_data_logits[:]
            loss_logits = F.mse_loss(self.model.forward(buffer_X), buffer_logits)

            buffer_X, buffer_labels = buffer_data_labels[:]
            loss_labels = self.loss_fn(self.model.forward(buffer_X), buffer_labels)

            final_loss = loss + alpha*loss_logits/len(self.training_loader) + beta*loss_labels/len(self.training_loader)
            
            final_loss.backward()

            train_loss += loss.item()
            #torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

        avg_train_loss = train_loss / len(self.training_loader)
        #print("\t-Average training loss: {0:.2f}".format(avg_train_loss))
        return avg_train_loss
    

    def train_with_norms(self):

        train_loss = 0
        self.model.train()
        for idx, batch in enumerate(self.training_loader):
            #if not self.mute:
                #print("\t### Batch {} out of {} ###".format(idx + 1, len(self.training_loader)))

            b_inputs = batch[0].to(self.device)
            b_labels = batch[1].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model.forward(b_inputs)
            loss = self.loss_fn(outputs, b_labels)
            loss.backward()
            for p in list(filter(lambda p: p.grad is not None, self.model.parameters())):
                p.grad.data.norm(2).item()

            train_loss += loss.item()
            #torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

        avg_train_loss = train_loss / len(self.training_loader)
        #print("\t-Average training loss: {0:.2f}".format(avg_train_loss))
        return avg_train_loss
    
    def train_datamap(self):
        conf = []
        train_loss = 0
        self.model.train()
        for idx, batch in enumerate(self.training_loader):
            #if not self.mute:
                #print("\t### Batch {} out of {} ###".format(idx + 1, len(self.training_loader)))

            b_inputs = batch[0].to(self.device)
            b_labels = batch[1].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model.forward(b_inputs)
            loss = self.loss_fn(outputs, b_labels)
            b_confs = select_probabilities(b_labels, outputs).tolist()
            conf.extend(b_confs)
            loss.backward()

            train_loss += loss.item()
            #torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

        avg_train_loss = train_loss / len(self.training_loader)
        #print("\t-Average training loss: {0:.2f}".format(avg_train_loss))
        return avg_train_loss, conf
    
    def curriculum_spotter(self, scores, epoch):
        losses = []

        for idx, batch in enumerate(self.test_loader):
            b_inputs = batch[0].to(self.device)
            b_labels = batch[1].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model.forward(b_inputs)
            b_y_pred = (outputs.detach().cpu().numpy() > self.threshold).astype(int)
            loss = self.loss_fn(outputs, b_labels)
            loss_samples = self.loss_per_sample(outputs, b_labels).mean(axis = 1)
            if idx == 0:
                all_outputs = outputs
                all_labels = b_labels
                losses = loss_samples
                preds = b_y_pred
            else:
                all_outputs = torch.cat((all_outputs, outputs), 0)
                all_labels = torch.cat((all_labels, b_labels), 0)
                losses = torch.cat((losses, loss_samples), 0)
                preds = np.vstack([preds, b_y_pred])

        all_labels = all_labels.detach().cpu().numpy()
        all_outputs = all_outputs.detach().cpu().numpy()
        losses = losses.detach().cpu().numpy()
        landa = compute_lambda(all_labels, preds, losses)
        easy_mask = _sample_easy(losses, landa)
        delta = epoch/100
        hard_mask = _sample_hard(landa, delta, losses)
        new_dataset_mask = easy_mask | hard_mask
        scores = _update_stat(hard_mask, losses)

        return scores, losses, new_dataset_mask
    
    def leitner_spotter(self, scores, epoch, queues, mapping):
        losses = []

        for idx, batch in enumerate(self.test_loader):
            b_inputs = batch[0].to(self.device)
            b_labels = batch[1].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model.forward(b_inputs)
            b_y_pred = (outputs.detach().cpu().numpy() > self.threshold).astype(int)
            loss = self.loss_fn(outputs, b_labels)
            loss_samples = self.loss_per_sample(outputs, b_labels).mean(axis = 1)
            if idx == 0:
                all_outputs = outputs
                all_labels = b_labels
                losses = loss_samples
                preds = b_y_pred
            else:
                all_outputs = torch.cat((all_outputs, outputs), 0)
                all_labels = torch.cat((all_labels, b_labels), 0)
                losses = torch.cat((losses, loss_samples), 0)
                preds = np.vstack([preds, b_y_pred])

        all_labels = all_labels.detach().cpu().numpy()
        all_outputs = all_outputs.detach().cpu().numpy()
        losses = losses.detach().cpu().numpy()
        queues = compute_new_queues(all_labels, preds, queues, mapping)
        scores = new_score_leitner(queues, losses, scores)

        return scores, losses, queues





    
    

