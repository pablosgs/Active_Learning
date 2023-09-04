import random
import numpy as np
import math
import torch
from sklearn.metrics import pairwise_distances
from scipy import stats
import pdb
import pandas as pd
import gc
from torch import nn
from src.classification.helpers import EarlyStopper
from torch.utils.data import DataLoader, TensorDataset
#from src.classification.plotting import plot_progress
import ranky as rk
from scipy.stats import entropy
from src.classification.models import DisModel
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from cleanlab.multilabel_classification.filter import find_label_issues


              
def random_query(preds, batch_size):
    indexes = random.sample(range(len(preds) - 1), batch_size)
    
    return indexes

def random_order(preds, batch_size):
    indexes = [*range(batch_size)]
    
    return indexes
    
def min_confidence(preds, batch_size):
    confidences = abs(0.5-preds)
    scores = np.min(confidences, axis = 1)
    max_idx = np.argpartition(scores, batch_size-1, axis=0)[:batch_size]
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

def mean_max_loss(preds, batch_size):
    class_pred = (preds > 0.5).astype(float)
    c_weight = 100 
    c_weights = torch.full([preds.shape[1]], c_weight)
    loss = torch.nn.BCELoss(reduction="none", weight=c_weights)
    losses = loss(torch.from_numpy(preds), torch.from_numpy(class_pred))
    mean_losses = torch.mean(losses, axis = 1).detach().cpu().numpy()
    max_idx = np.argpartition(-mean_losses, batch_size-1, axis=0)[:batch_size]
    return max_idx

def max_loss(preds, class_pred, batch_size):
    #class_pred = (preds > 0.5).astype(float)
    c_weight = 100 
    c_weights = torch.full([preds.shape[1]], c_weight)
    loss = torch.nn.BCELoss(reduction="none", weight=c_weights)
    losses = loss(torch.from_numpy(preds), torch.from_numpy(class_pred))
    mean_losses = torch.max(losses, dim = 1).values.detach().cpu().numpy()
    max_idx = np.argpartition(-mean_losses, batch_size-1, axis=0)[:batch_size]
    return max_idx

def label_cardinality(preds, batch_size):
    class_pred = (preds > 0.5).astype(int)
    scores = np.square(np.transpose(class_pred)[0] - np.mean(np.transpose(class_pred)[0])) + np.square(np.transpose(class_pred)[1] - np.mean(np.transpose(class_pred)[1])) + np.square(np.transpose(class_pred)[2] - np.mean(np.transpose(class_pred)[2])) + np.square(np.transpose(class_pred)[3] - np.mean(np.transpose(class_pred)[3])) + np.square(np.transpose(class_pred)[4] - np.mean(np.transpose(class_pred)[4])) + np.square(np.transpose(class_pred)[5] - np.mean(np.transpose(class_pred)[5]))
    max_idx = np.argpartition(-scores, batch_size-1, axis=0)[:batch_size]
    return max_idx

def score_label_card(preds, batch_size, beta):
    class_pred = (preds > 0.5).astype(int)
    scores_label = np.square(np.transpose(class_pred)[0] - np.mean(np.transpose(class_pred)[0])) + np.square(np.transpose(class_pred)[1] - np.mean(np.transpose(class_pred)[1])) + np.square(np.transpose(class_pred)[2] - np.mean(np.transpose(class_pred)[2])) + np.square(np.transpose(class_pred)[3] - np.mean(np.transpose(class_pred)[3])) + np.square(np.transpose(class_pred)[4] - np.mean(np.transpose(class_pred)[4])) + np.square(np.transpose(class_pred)[5] - np.mean(np.transpose(class_pred)[5]))/np.mean(np.mean(np.transpose(class_pred)))
    scores = -preds*np.log2(preds) - (1-preds)*np.log2(1-preds)
    classwise_max = np.mean(scores, axis=1)
    final_score = (classwise_max**beta)*(scores_label**(1-beta))
    max_idx = np.argpartition(-final_score, batch_size-1, axis=0)[:batch_size]
    return max_idx

def badge(preds, embeddings, batch_size): #https://github.com/forest-snow/alps/blob/main/src/sample.py#L78
    class_pred = (preds > 0.5).astype(float)
    scales = preds - class_pred
    c_weight = 100 
    c_weights = torch.full([preds.shape[1]], c_weight)
    #loss = torch.nn.BCELoss(reduction="none", weight=c_weights)
    #scales = loss(torch.from_numpy(preds), torch.from_numpy(class_pred))
    unlab_size, label_size = scales.shape[0], scales.shape[1]
    grads_3d = torch.einsum('bi,bj->bij', torch.from_numpy(scales), torch.from_numpy(embeddings))
    grads = grads_3d.view(unlab_size, -1)
    centers = kmeans_pp(grads, batch_size, [])
    return centers

def kmeans_pp(X, k, centers, **kwargs):
    # kmeans++ algorithm
    if len(centers) == 0:
        # randomly choose first center
        c1 = np.random.choice(X.size(0))
        centers.append(c1)
        k -= 1
    # greedily choose centers
    for i in range(k):
        dist = closest_center_dist(X, centers) ** 2
        prob = (dist / dist.sum()).cpu().detach().numpy()
        ci = np.random.choice(X.size(0), p=prob)
        centers.append(ci)
    return centers

def closest_center_dist(X, centers):
    # return distance to closest center
    dist = torch.cdist(X, X[centers])
    cd = dist.min(axis=1).values
    return cd

def expected_gradient(preds, embeddings, batch_size):
    class_pred = (preds > 0.5).astype(float)
    #scales = preds - class_pred
    c_weight = 100 
    c_weights = torch.full([preds.shape[1]], c_weight)
    loss = torch.nn.BCELoss(reduction="none", weight=c_weights)
    scales = loss(torch.from_numpy(preds), torch.from_numpy(class_pred))
    unlab_size, label_size = scales.shape[0], scales.shape[1]
    grads_3d = torch.einsum('bi,bj->bij', scales, torch.from_numpy(embeddings))
    grads = grads_3d.view(unlab_size, -1)
    scores = np.abs(grads.sum(axis=1).numpy())
    max_idx = np.argpartition(-scores, batch_size-1, axis=0)[:batch_size]
    return max_idx

def cvirs(pred, data, batch_size):
    [X_lab, X_unlab, X_test, y_lab, y_unlab, y_test, labels] = data
    margin = pd.DataFrame(np.abs(2*pred - 1))
    label_ranks = pd.DataFrame(columns=['0', '1', '2', '3', '4', '5', '6'])
    for i in range(7):
        label_ranks[str(i)] = margin[i].rank(ascending=False)
    ranks = rk.borda(label_ranks)

    similarity_matrix = csm(X_unlab, X_lab)
    score_similarity = similarity_matrix.sum(axis = 1) #sum over rows I think
    scores = np.multiply(ranks, score_similarity)
    max_idx = np.argpartition(scores, batch_size-1, axis=0)[:batch_size]
    return max_idx

def alucs(pred, data, batch_size,beta):
    [X_lab, X_unlab, X_test, y_lab, y_unlab, y_test, labels] = data
    class_pred = (pred > 0.5).astype(int)
    scores_label = np.square(np.transpose(class_pred)[0] - np.mean(np.transpose(y_test)[0])) + np.square(np.transpose(class_pred)[1] - np.mean(np.transpose(y_test)[1])) + np.square(np.transpose(class_pred)[2] - np.mean(np.transpose(y_test)[2])) + np.square(np.transpose(class_pred)[3] - np.mean(np.transpose(y_test)[3])) + np.square(np.transpose(class_pred)[4] - np.mean(np.transpose(y_test)[4])) + np.square(np.transpose(class_pred)[5] - np.mean(np.transpose(y_test)[5]))/np.mean(np.mean(np.transpose(y_test)))
    final_label = scores_label**(1-beta)

    

    margin = pd.DataFrame(np.abs(2*pred - 1))
    label_ranks = pd.DataFrame(columns=['0', '1', '2', '3', '4', '5', '6'])
    for i in range(7):
        label_ranks[str(i)] = margin[i].rank(ascending=False)
    ranks = rk.borda(label_ranks).to_numpy()**beta

    similarity_matrix = csm(X_unlab, X_lab)
    score_similarity = (1/similarity_matrix.sum(axis = 1))**(1-beta) #sum over rows I think
    #scores = np.multiply.reduce((np.sqrt(ranks/np.linalg.norm(ranks)), score_similarity/np.linalg.norm(score_similarity), final_label/np.linalg.norm(final_label)))
    scores = np.multiply(ranks,np.multiply(score_similarity, final_label))
    max_idx = np.argpartition(-scores, batch_size-1, axis=0)[:batch_size]
    return max_idx

def dalucs(pred, data, batch_size,beta):
    [X_lab, X_unlab, X_test, y_lab, y_unlab, y_test, labels] = data
    class_pred = (pred > 0.5).astype(int)
    #scores_label = np.square(np.transpose(class_pred)[0] - np.mean(np.transpose(y_lab)[0])) + np.square(np.transpose(class_pred)[1] - np.mean(np.transpose(y_lab)[1])) + np.square(np.transpose(class_pred)[2] - np.mean(np.transpose(y_lab)[2])) + np.square(np.transpose(class_pred)[3] - np.mean(np.transpose(y_lab)[3])) + np.square(np.transpose(class_pred)[4] - np.mean(np.transpose(y_lab)[4])) + np.square(np.transpose(class_pred)[5] - np.mean(np.transpose(y_lab)[5]))/np.mean(np.mean(np.transpose(y_lab)))
    #final_label = scores_label**(1-beta)

    margin = pd.DataFrame(np.abs(2*pred - 1))
    label_ranks = pd.DataFrame(columns=['0', '1', '2', '3', '4', '5', '6'])
    for i in range(6):
        label_ranks[str(i)] = margin[i].rank(ascending=False)
    ranks = rk.borda(label_ranks).to_numpy()**beta

    similarity_matrix = csm(X_unlab, X_lab)
    score_similarity = (1/similarity_matrix.sum(axis = 1))**(1-beta) #sum over rows I think

    #score_l = np.multiply(ranks, final_label)
    score_s = np.multiply(ranks, score_similarity)
    #dt = pd.DataFrame({'label': score_l, 'sim': score_s})
    #dt['l_rank'] = dt['label'].rank(ascending=False)
    #dt['s_rank'] = dt['sim'].rank(ascending=False)
    #scores = rk.pairwise(dt[['l_rank','s_rank']])
    #scores = rk.borda(dt).to_numpy()
    #scores = np.multiply.reduce((np.sqrt(ranks/np.linalg.norm(ranks)), score_similarity/np.linalg.norm(score_similarity), final_label/np.linalg.norm(final_label)))
    max_idx = np.argpartition(-score_s, batch_size-1, axis=0)[:batch_size]
    return max_idx

def borda(pred, batch_size):

    margin = pd.DataFrame(np.abs(2*pred - 1))
    label_ranks = pd.DataFrame(columns=['0', '1', '2', '3', '4', '5', '6'])
    for i in range(7):
        label_ranks[str(i)] = margin[i].rank(ascending=False)
    scores = rk.borda(label_ranks).to_numpy()
    max_idx = np.argpartition(-scores, batch_size-1, axis=0)[:batch_size]
    return max_idx

def csm(A,B): #https://towardsdatascience.com/cosine-similarity-matrix-using-broadcasting-in-python-2b1998ab3ff3
    num=np.dot(A,B.T)
    p1=np.sqrt(np.sum(A**2,axis=1))[:,np.newaxis]
    p2=np.sqrt(np.sum(B**2,axis=1))[np.newaxis,:]
    return num/(p1*p2)

def DAL(data, batch_size):
    [X_lab, X_unlab, X_test, y_lab, y_unlab, y_test, labels] = data
    labeled_so_far = 0
    sub_sample_size = int(batch_size/1)
    selected_indices = []
    count = 0
    while labeled_so_far < batch_size:
        if labeled_so_far + sub_sample_size > batch_size:
            sub_sample_size = batch_size - labeled_so_far

        model, training_stats = train_discriminative_model(X_lab, X_unlab)
        predictions = model.forward(torch.Tensor(X_unlab))
        selected_indices.append(np.argpartition(-predictions.cpu().detach().numpy().reshape(predictions.shape[0],), sub_sample_size-1)[:sub_sample_size].tolist())
        labeled_so_far += sub_sample_size

        # delete the model to free GPU memory:

        del model
        gc.collect()
    selected_indices = np.array(selected_indices).squeeze()
    return selected_indices

def train_discriminative_model(labeled, unlabeled, gpu=1):
    """
    A function that trains and returns a discriminative model on the labeled and unlabaled data.
    """

    # create the binary dataset:
    y_L = np.zeros((labeled.shape[0],1),dtype='int')
    y_U = np.ones((unlabeled.shape[0],1),dtype='int')
    X_train = torch.Tensor(np.vstack((labeled, unlabeled)))
    Y_train = np.append(y_L, y_U)
    Y_train = torch.Tensor(Y_train)[:,None]

    # build the model:
    weights = torch.FloatTensor([float(X_train.shape[0]) / Y_train[Y_train==0].shape[0], float(X_train.shape[0]) / Y_train[Y_train==1].shape[0]])
    input_layer_size = 384
    output_layer_size = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DisModel(input_layer_size, output_layer_size, device)
    
    # train the model:
    batch_size = 1024
    loss_fn = torch.nn.BCELoss(reduction="mean", weight = torch.FloatTensor([float(X_train.shape[0]) / Y_train[Y_train==1].shape[0]])) 
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    epochs = 100
    early_stopper = EarlyStopper()
    model.train()
    training_stats = []
    for t in range(epochs):
        train_loss = 0
        dataset = TensorDataset(X_train, Y_train)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for id_batch, (x_batch, y_batch) in enumerate(dataloader):

            y_batch_pred = model(x_batch)

            loss = loss_fn(y_batch_pred, y_batch)
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_train_loss = train_loss / len(dataloader)
        training_stats.append({"epoch": t + 1, "Training Loss": avg_train_loss})

    return model, training_stats

def clue(embeddings, preds, batch_size):

    scores = -preds*np.log2(preds) - (1-preds)*np.log2(1-preds)
    mean_ent = np.mean(scores, axis=1)
    km = KMeans(batch_size)
    km.fit(embeddings, sample_weight=mean_ent)
    dists = euclidean_distances(km.cluster_centers_, embeddings)
    sort_idxs = dists.argsort(axis=1)
    q_idxs = []
    ax, rem = 0, batch_size
    while rem > 0:
        q_idxs.extend(list(sort_idxs[:, ax][:rem]))
        q_idxs = list(set(q_idxs))
        rem = batch_size-len(q_idxs)
        ax += 1
    return q_idxs
      
######### ACTIVE LABEL CORRECTION ##############


def alc_mislabeled(preds, data, batch_size):
    [X_lab, X_unlab, X_test, y_lab, y_unlab, y_test, labels] = data
    diff = y_lab - preds
    classwise_max = np.max(diff, axis=1)
    max_idx = np.argpartition(-classwise_max, batch_size-1, axis=0)[:batch_size]
    return max_idx

def alc_disagreement(preds, batch_size):
    max_idx = max_entropy(preds, batch_size)
    return max_idx

def self_confidence(preds, data, batch_size):
    return find_label_issues(
    labels=data[3],
    pred_probs=preds,
    return_indices_ranked_by="self_confidence",)