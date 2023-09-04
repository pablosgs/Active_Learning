import random
import json

import matplotlib.pyplot as plt
import numpy as np
import scipy
import pandas as pd

from tqdm.auto import tqdm


def plot_progress(epochs, training_loss, validation_loss):

    fig, ax = plt.subplots(figsize=(16, 9)) #lmvfd

    color = 'tab:red'
    ax.plot(epochs, training_loss, color=color, label='Training loss')
    ax.tick_params(axis='y', labelcolor=color, labelsize=16)

    color = 'tab:blue'
    right_y = ax.twinx()
    right_y.plot(epochs, validation_loss, color=color, label='Validation loss')
    right_y.tick_params(axis='y', labelcolor=color, labelsize=16)

    ax.set_xlabel('epoch', fontsize=16)
    ax.tick_params(axis='x', labelsize=16)

    fig.tight_layout()
    ax.grid(True)
    fig.legend(loc='center right', fontsize=20, bbox_to_anchor=(0.925, 0.535))
    plt.savefig('discriminative.png')

def plot_active_process(progress):
#def plot_active_process(nr_samples, final_acc, final_f1, final_loss):

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Active learning progress')
    ax1.plot(progress['nr_samples'], progress['final_acc'], label = 'Accuracy')
    ax1.plot(progress['nr_samples'], progress['final_f1'], label = 'F1')
    ax1.set_title('F1 and precision')
    ax1.set(xlabel='Nr samples', ylabel='Performance')
    ax1.legend()
    ax2.plot(progress['nr_samples'], progress['final_loss'], label = 'Loss')
    ax2.set(xlabel='Nr samples', ylabel='Loss')
    ax2.set_title('Loss')
    ax2.legend()
    plt.show()

def plot_active_2(progress):
#def plot_active_process(nr_samples, final_acc, final_f1, final_loss):

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Active learning progress')
    ax1.plot(progress['nr_samples'], progress['final_acc'], label = 'Accuracy')
    ax1.plot(progress['nr_samples'], progress['final_f1'], label = 'F1')
    ax1.set_title('F1 and precision')
    ax1.set(xlabel='Nr samples', ylabel='Performance')
    ax1.legend()
    ax2.plot(progress['nr_samples'], progress['final_loss'], label = 'Loss')
    ax2.set(xlabel='Nr samples', ylabel='Loss')
    ax2.set_title('Loss')
    ax2.legend()
    plt.show()

def plot_continual_train_loss():

    with open('/home/pablo/active-learning-pablo/results/test/continual_learning/dark_wo_duplicates/retrain.json', 'r') as f:
        loss = json.load(f)

    final_loss = []
    loss_trial = {}
    count = 0
    for trial in loss:
        count += 1
        training_losses = []
        epoch_nr = []
        for task in trial:
            for epoch in task:
                training_losses.append(epoch['Training Loss'])
                epoch_nr.append(epoch['epoch'])
        loss_trial[str(count)] = training_losses

    df_loss = pd.DataFrame.from_dict(loss_trial)
    final_loss1 = df_loss.mean(axis=1)
    final_loss1_std = df_loss.std(axis=1)


    with open('/home/pablo/active-learning-pablo/results/test/continual_learning/dark_wo_duplicates/warm.json', 'r') as f:
        loss = json.load(f)

    final_loss = []
    loss_trial = {}
    count = 0
    for trial in loss:
        count += 1
        training_losses = []
        epoch_nr = []
        for task in trial:
            for epoch in task:
                training_losses.append(epoch['Training Loss'])
                epoch_nr.append(epoch['epoch'])
        loss_trial[str(count)] = training_losses

    df_loss = pd.DataFrame.from_dict(loss_trial)
    final_loss2 = df_loss.mean(axis=1)
    final_loss2_std = df_loss.std(axis=1)

    with open('/home/pablo/active-learning-pablo/results/test/continual_learning/dark_wo_duplicates/shrink.json', 'r') as f:
        loss = json.load(f)

    final_loss = []
    loss_trial = {}
    count = 0
    for trial in loss:
        count += 1
        training_losses = []
        epoch_nr = []
        for task in trial:
            for epoch in task:
                training_losses.append(epoch['Training Loss'])
                epoch_nr.append(epoch['epoch'])
        loss_trial[str(count)] = training_losses

    df_loss = pd.DataFrame.from_dict(loss_trial)
    final_loss3 = df_loss.mean(axis=1)
    final_loss3_std = df_loss.std(axis=1)

    with open('/home/pablo/active-learning-pablo/results/test/continual_learning/dark_wo_duplicates/lwf.json', 'r') as f:
        loss = json.load(f)

    final_loss = []
    loss_trial = {}
    count = 0
    for trial in loss:
        count += 1
        training_losses = []
        epoch_nr = []
        for task in trial:
            for epoch in task:
                training_losses.append(epoch['Training Loss'])
                epoch_nr.append(epoch['epoch'])
        loss_trial[str(count)] = training_losses

    df_loss = pd.DataFrame.from_dict(loss_trial)
    final_loss4 = df_loss.mean(axis=1)
    final_loss4_std = df_loss.std(axis=1)

    with open('/home/pablo/active-learning-pablo/results/test/continual_learning/dark_wo_duplicates/ewc.json', 'r') as f:
        loss = json.load(f)

    final_loss = []
    loss_trial = {}
    count = 0
    for trial in loss:
        count += 1
        training_losses = []
        epoch_nr = []
        for task in trial:
            for epoch in task:
                training_losses.append(epoch['Training Loss'])
                epoch_nr.append(epoch['epoch'])
        loss_trial[str(count)] = training_losses

    df_loss = pd.DataFrame.from_dict(loss_trial)
    final_loss5 = df_loss.mean(axis=1)
    final_loss5_std = df_loss.std(axis=1)

    """"
    with open('results/test/continual_learning/cold.json', 'r') as f:
        loss = json.load(f)

    final_loss = []
    loss_trial = {}
    count = 0
    for trial in loss:
        count += 1
        training_losses = []
        epoch_nr = []
        for task in trial:
            for epoch in task:
                training_losses.append(epoch['Training Loss'])
                epoch_nr.append(epoch['epoch'])
        loss_trial[str(count)] = training_losses

    df_loss = pd.DataFrame.from_dict(loss_trial)
    final_loss5 = df_loss.mean(axis=1)
    final_loss5_std = df_loss.std(axis=1)
    """
    
    fig, ax = plt.subplots(figsize=(16, 9)) #lmvfd

    color = 'tab:red'
    ax.plot(final_loss1, color=color, label='Retraining')
    ax.plot(final_loss2, color='tab:blue', label='Warm-start')
    ax.plot(final_loss3, color='tab:green', label='Shrink')
    ax.plot(final_loss4, color='m', label='Self-distillation')
    #ax.plot(final_loss5, color='lightseagreen', label='EWC')
    #ax.plot(final_loss5, color='y', label='Cold')

    ax.tick_params(axis='y', labelsize=16)
    ax.tick_params(axis='x', labelsize=16)

    fig.tight_layout()
    fig.legend(loc='upper right', fontsize=20)
    plt.savefig('prueba.png')

plot_continual_train_loss()

""""
def csm(A,B): #https://towardsdatascience.com/cosine-similarity-matrix-using-broadcasting-in-python-2b1998ab3ff3
    num=np.dot(A,B.T)
    p1=np.sqrt(np.sum(A**2,axis=1))[:,np.newaxis]
    p2=np.sqrt(np.sum(B**2,axis=1))[np.newaxis,:]
    return num/(p1*p2)

def find_similarity(data):
    matrix = csm(data, data)
    np.save('test3.npy', data)
"""