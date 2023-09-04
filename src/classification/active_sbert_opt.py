import os
import json
import torch
import numpy as np
from src.classification.helpers import EarlyStopper
from tqdm.auto import tqdm
import pandas as pd

from src.classification.models import SBERTModel
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, TensorDataset
from src.classification.helpers import split_train_val_test
import torchcontrib
from sklearn.metrics import accuracy_score
import torch.optim.swa_utils as swa


        
        

def fit(trial):
    params = {"lr": trial.suggest_categorical('lr',[1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3]), "epoch": trial.suggest_categorical('epoch', [50, 70, 100,150]), "dropout": trial.suggest_categorical('dropout',[0, 0.1, 0.2, 0.5]), "decay": trial.suggest_float('decay', 0, 0.2), "hidden": trial.suggest_categorical('hidden',[50, 100, 200, 300]), "batch": trial.suggest_categorical('batch',[32, 64, 128])}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    labels = [
        'Cybercrime',
       'Drugs / Narcotics', 'Financial Crime', 'Goods and Services',
       'Sexual Abuse', 'Violent Crime']
    df = pd.read_csv('/home/pablo/active-learning-pablo/data/datasets/cflw_final.csv')
    y = df.iloc[:, -len(labels) :].values
    X = df.iloc[:, : -len(labels)].values

    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(
        X, y, train_frac=0.7, val_frac=0.2, test_frac=0.1
    )
    # train_df = pd.read_csv('/home/pablo/active-learning-pablo/data/datasets/toxic_train_vec_final.csv')
    # test_df = pd.read_csv('/home/pablo/active-learning-pablo/data/datasets/toxic_test_vec_final.csv')
    # val_df = pd.read_csv('/home/pablo/active-learning-pablo/data/datasets/toxic_val_vec_final.csv')
    # labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate', 'non_toxic']

    # val_toxic = val_df[val_df[val_df.columns.values[-1]] == 0]

    # train_df = pd.concat([train_df,val_toxic])

    # #train_df.drop(columns=['390'], inplace=True)
    # #test_df.drop(columns=['390'], inplace=True)
    # #val_df.drop(columns=['390'], inplace=True)
    
    # X_train = train_df[train_df.columns.values[:384]].values#X_train = df_train[df_train.columns.values[:384]].values
    # y_train = train_df[train_df.columns.values[384:]].values #y_train = df_train[df_train.columns.values[384:]].values

    # X_test = test_df[test_df.columns.values[:384]].values #X_test = df_test[df_test.columns.values[:384]].values
    # y_test = test_df[test_df.columns.values[384:]].values #y_test = df_test[df_test.columns.values[384:]].values

    # X_val = val_df[val_df.columns.values[:384]].values #X_test = df_test[df_test.columns.values[:384]].values
    # y_val = val_df[val_df.columns.values[384:]].values #y_test = df_test[df_test.columns.values[384:]].values
    
    data = [X_train, X_val, X_test, y_train, y_val, y_test, labels]
    #data = [X_train, X_test, y_train, y_test, labels]

    training_stats = []
    early_stopper = EarlyStopper()
    #min_val_loss = np.inf
    # Setup datasets
    X_lab, X_unlab, X_test, y_lab, y_unlab, y_test, labels = data
    batch_test = X_test.shape[0]
    batch_size = params["batch"]
    threshold = 0.5
    torch_weights = torch.from_numpy((len(y_train)-np.sum(y_train,axis=0))/np.sum(y_train,axis=0))
    

    training_data = TensorDataset(
        torch.from_numpy(X_train.astype(np.float32)), torch.from_numpy(y_train.astype(np.float32))
    )
    unlabeled_data = TensorDataset(
        torch.from_numpy(X_val.astype(np.float32)), torch.from_numpy(y_val.astype(np.float32))
    )

    test_data = TensorDataset(
        torch.from_numpy(X_test.astype(np.float32)), torch.from_numpy(y_test.astype(np.float32))
    )

    training_loader = DataLoader(training_data, batch_size=batch_size)
    unlabeled_loader = DataLoader(unlabeled_data, batch_size=X_val.shape[0])
    validation_loader = DataLoader(unlabeled_data, batch_size=X_val.shape[0])
    test_loader = DataLoader(test_data, batch_size=X_test.shape[0])
    

    # Setup model
    input_layer_size = 384
    hidden_layer_size = params["hidden"]
    output_layer_size = 6
    dropout = params["dropout"]
    #c_weight = params['c_weight']  # ​>1 increases the recall, ​<1 increases the precision
    lr = params["lr"]
    wd = params['decay']
    n_epochs = params["epoch"]
    model = SBERTModel(input_layer_size, hidden_layer_size, output_layer_size, device, dropout)
    #c_weights = torch.full([len(labels)], c_weight)
    #self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch_weights, reduction="mean", weight=c_weights)  
    loss_fn = torch.nn.BCELoss(reduction="mean", weight=torch_weights) 
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)


    for epoch in range(n_epochs):

        train_loss = 0
        model.train()

        for idx, batch in enumerate(training_loader):

            b_inputs = batch[0].to(device)
            b_labels = batch[1].to(device)

            optimizer.zero_grad()
            outputs = model.forward(b_inputs)
            loss = loss_fn(outputs, b_labels)
            loss.backward()

            train_loss += loss.item()
            
            optimizer.step()

        avg_train_loss = train_loss / len(training_loader)

        ### VALIDATION
        model.eval()
        val_loss = 0
        for idx, batch in enumerate(validation_loader):
            b_inputs = batch[0].to(device)
            b_labels = batch[1].to(device)

            with torch.no_grad():
                outputs = model.forward(b_inputs)
                #print(torch.mean(torch.mean(outputs)))
                loss = loss_fn(outputs, b_labels)

            val_loss += loss.item()

        avg_val_loss = val_loss / len(validation_loader)

    y_pred = np.empty(shape=[0, len(labels)])
    y_true = np.empty(shape=[0, len(labels)])

    model.eval()
    for batch in test_loader:
        b_inputs = batch[0].to(device)
        b_labels = batch[1].to(device)

        with torch.no_grad():
            b_y_pred = model.forward(b_inputs)
            b_y_pred = (b_y_pred.cpu().numpy() > threshold).astype(int)
        y_pred = np.append(y_pred, b_y_pred, axis=0)
        y_true = np.append(y_true, b_labels, axis=0)

    results = classification_report(y_true, y_pred, output_dict=True, target_names=labels, zero_division=False)

    micro_f1 = results['micro avg']['f1-score']
    macro_f1 = results['macro avg']['f1-score']
    #tune.report(mean_accuracy = f1)
    
    return micro_f1, macro_f1



def save_model(self, model_save_path: str) -> None:
    torch.save(self.model.state_dict(), model_save_path)


