import os
import json
import torch
import numpy as np
from src.classification.helpers import EarlyStopper

from src.classification.models import SBERTModel
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, TensorDataset


class ClassificationManagerSBERT:
    def __init__(self, data, model_save_path: str, overwrite_model: bool, mute: bool) -> list:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model_save_path = model_save_path
        if not overwrite_model and os.path.exists(model_save_path):
            raise FileExistsError("A model with this name already exists")

        self.mute = mute
        batch_size = 32
        self.threshold = 0.5

        # Setup datasets
        X_train, X_val, X_test, y_train, y_val, y_test, labels = data
        self.labels = labels

        training_data = TensorDataset(
            torch.from_numpy(X_train.astype(np.float32)), torch.from_numpy(y_train.astype(np.float32))
        )
        validation_data = TensorDataset(
            torch.from_numpy(X_val.astype(np.float32)), torch.from_numpy(y_val.astype(np.float32))
        )
        test_data = TensorDataset(
            torch.from_numpy(X_test.astype(np.float32)), torch.from_numpy(y_test.astype(np.float32))
        )

        self.training_loader = DataLoader(training_data, batch_size=batch_size)
        self.validation_loader = DataLoader(validation_data, batch_size=batch_size)
        self.test_loader = DataLoader(test_data, batch_size=batch_size)

        # Setup model
        input_layer_size = 384
        hidden_layer_size = 50
        output_layer_size = 6
        dropout = 0.5
        c_weight = 0.1  # ​>1 increases the recall, ​<1 increases the precision
        lr = 1e-5
        wd = 0.1
        self.model = SBERTModel(input_layer_size, hidden_layer_size, output_layer_size, self.device, dropout)
        c_weights = torch.full([len(labels)], c_weight)
        self.loss_fn = torch.nn.BCELoss(weight=c_weights, reduction="sum")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)

    def train_model(self, n_epochs=10):
        training_stats = []
        early_stopper = EarlyStopper()
        min_val_loss = np.inf

        for epoch in range(n_epochs):
            print("\n======== Epoch {:} / {:} ========".format(epoch + 1, n_epochs))

            print("######## Epoch {}: Training Start ########".format(epoch + 1))
            avg_train_loss = self.train_epoch()

            print("######## Epoch {}: Validation Start ########".format(epoch + 1))
            avg_val_loss = self.validate_epoch()

            training_stats.append({"epoch": epoch + 1, "Training Loss": avg_train_loss, "Valid. Loss": avg_val_loss})

            if early_stopper.early_stop(avg_val_loss):
                break

        if avg_val_loss <= min_val_loss:
            self.save_model(self.model_save_path)
        

        return training_stats

    def train_epoch(self):
        train_loss = 0
        self.model.train()
        for idx, batch in enumerate(self.training_loader):
            if not self.mute:
                print("\t### Batch {} out of {} ###".format(idx + 1, len(self.training_loader)))

            b_inputs = batch[0].to(self.device)
            b_labels = batch[1].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model.forward(b_inputs)
            loss = self.loss_fn(outputs, b_labels)
            loss.backward()

            train_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

        avg_train_loss = train_loss / len(self.training_loader)
        print("\t-Average training loss: {0:.2f}".format(avg_train_loss))
        return avg_train_loss

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
                loss = self.loss_fn(outputs, b_labels)

            val_loss += loss.item()

        avg_val_loss = val_loss / len(self.validation_loader)
        print("\t-Average validation loss: {0:.2f}".format(avg_val_loss))
        return avg_val_loss

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

        results = classification_report(y_true, y_pred, output_dict=True, target_names=self.labels, zero_division=False)
        return results

    def save_model(self, model_save_path: str) -> None:
        torch.save(self.model.state_dict(), model_save_path)
