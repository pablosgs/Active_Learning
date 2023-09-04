import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.classification.helpers import split_train_val_test
from src.classification.active_sbert import ClassificationManagerSBERT
#from src.classification.plotting import plot_active_process
from actve_learning.query_strategies import random_query, random_order, min_confidence, max_score, mean_entropy, max_entropy, label_cardinality, mean_max_loss, badge, score_label_card, expected_gradient, DAL, cvirs, alucs, dalucs, clue
import torch 
import random
import json
from copy import deepcopy
import time

def shrink_perturb(model, lamda=0.5, sigma=0.01):
    for (name, param) in model.named_parameters():
        if 'weight' in name:   # just weights
            nc = param.shape[0]  # cols
            nr = param.shape[1]  # rows
            for i in range(nr):
                for j in range(nc):
                    param.data[j][i] = \
                        (lamda * param.data[j][i]) + \
                        torch.normal(0.0, sigma, size=(1,1))
    return

def pass_data(data, indexes):

    [X_lab, X_unlab, X_test, y_lab, y_unlab, y_test, labels] = data
    extracted_X = X_unlab[indexes,:]
    extracted_y = y_unlab[indexes,:]
    X_lab = np.vstack((X_lab, extracted_X))
    y_lab = np.vstack((y_lab, extracted_y))
    X_unlab = np.delete(X_unlab, indexes, axis=0)
    y_unlab = np.delete(y_unlab, indexes, axis=0)
    data = [extracted_X, X_unlab, X_test, extracted_y, y_unlab, y_test, labels]
    return data

def extend_buffer(buffer_data, data, indexes, preds):
    class_pred = (preds > 0.5).astype(int)
    [X_lab, X_unlab, X_test, y_lab, y_unlab, y_test, labels] = data
    first = False
    if buffer_data['X'].shape[0] < 2:
        first = True
    extracted_X = X_lab[indexes,:]
    extracted_labels = y_lab[indexes,:]
    extracted_logits = class_pred[indexes,:]
    buffer_data['X'] = np.vstack((buffer_data['X'],extracted_X))
    buffer_data['labels'] = np.vstack((buffer_data['labels'],extracted_labels))
    buffer_data['logits'] = np.vstack((buffer_data['logits'], extracted_logits))
    if first:
        buffer_data['X'] = np.delete(buffer_data['X'], (0), axis=0)
        buffer_data['labels'] = np.delete(buffer_data['labels'], (0), axis=0)
        buffer_data['logits'] = np.delete(buffer_data['logits'], (0), axis=0)
    return buffer_data

def store_results(results, accuracy, progress, data):
    micro_f1 = results['micro avg']['f1-score']
    macro_f1 = results['macro avg']['f1-score']
    progress['nr_samples'].append(len(data[0]))
    progress['final_acc'].append(accuracy)
    progress['final_micro_f1'].append(micro_f1)
    progress['final_macro_f1'].append(macro_f1)
    progress['results'] = results
    return progress

def create_buffer(data, size):
    [X_lab, X_unlab, X_test, y_lab, y_unlab, y_test, labels] = data


def main():


    labels = [
        'Cybercrime',
       'Drugs / Narcotics', 'Financial Crime', 'Goods and Services',
       'Sexual Abuse', 'Violent Crime']
    df = pd.read_csv('/home/pablo/active-learning-pablo/data/datasets/cflw_large2_vec.csv')
    #financial_df = df[df['386'] == 1]
    #non_financial_df = df[df['386'] == 0]
    #df = pd.concat([non_financial_df,financial_df.sample(10_000)])
    y = df.iloc[:, -len(labels) :].values
    X = df.iloc[:, : -len(labels)].values

    X_unlab, X_lab, X_test, y_unlab, y_lab, y_test = split_train_val_test(
        X, y, train_frac=0.7, val_frac=0.1, test_frac=0.2
    )
    data = [X_lab, X_unlab, X_test, y_lab, y_unlab, y_test, labels]

    # Query strategy = uncertainity
    """"
    Start Humnan-in-the-loop process. This loop consists of thw following iterative process:

    1. Initialize and train the model with the labeled data.
    2. Evaluate performance of the model and store.
    3. Make predictions with unlabeled data.
    4. Query most important samples based on the predictions.
    5. Pass sampled instances (S) from unlabeled (U) to labeled (L) dataset -> L = L + S, U = U - S
    6. Start loop again if condition is not fulfiled (by the moment just for loop)
    """

    progress = {}
    progress['nr_samples'] = []
    progress['final_acc'] = []
    progress['final_micro_f1'] = []
    progress['final_macro_f1'] = []
    progress['final_loss'] = []
    progress['time'] = []

    beta = 1
    shrink = 1
    perturb = 0.1
    overall_loss = []
    buffer_data = {}
    buffer_data['X'] = np.zeros(384)
    buffer_data['labels'] = np.zeros(6)
    buffer_data['logits'] = np.zeros(6)


    for i in tqdm(range(5), leave = True): #Add stopping criteria
        
        #1. Train model
        start = time.time()
        model = ClassificationManagerSBERT(data, mute=True)
        if i >= 1:
            teacher = ClassificationManagerSBERT(data, mute=True)
            teacher.model.load_state_dict(torch.load('/home/pablo/active-learning-pablo/results/test/continual_learning/models/teacher_model.pth'))
            #model.model.load_state_dict(torch.load('/home/pablo/active-learning-pablo/results/test/continual_learning/models/teacher_model.pth'))
            loss = model.fit(teacher = teacher, n_epochs = 100, derpp = 1, buffer = buffer_data)
        else:
            loss = model.fit(n_epochs = 100)
        pred_lab = np.zeros(y_lab.shape)
        index_buffer = random_query(pred_lab, int(0.1*pred_lab.shape[0]))
        buffer_data = extend_buffer(buffer_data, data, index_buffer, pred_lab)
        #overall_loss['Task ' + str(i)] = loss
        overall_loss.append(loss)
        torch.save(model.model.state_dict(), '/home/pablo/active-learning-pablo/results/test/continual_learning/models/teacher_model.pth')
        end = time.time()
        progress['time'].append(end-start)

        #2. Evaluate model and save results
        results, accuracy = model.evaluate_model()
        progress = store_results(results, accuracy, progress, data)
    

        #3. Make predictions on U
        pred = model.predict_unlab()


        #4. Query
        batch_size = 2000
        if i < 4:
            indexes = random_query(pred, batch_size)
        #indexes = clue(embeddings, pred, batch_size)

        #5. Pass sampled instances
            data = pass_data(data, indexes)

        #shrink = 0.5
        #perturb = 0.001

        

    #plot_active_process(nr_samples, final_acc, final_f1, final_loss)
    #plot_active_process(progress)
    #print(pred)
    #print('Final accuracy: ', results['samples avg']['precision'])

    return progress, overall_loss

if __name__ == "__main__":
    
    
    
    data = {str(i): [] for i in range(5)}
    final_progress = {}
    final_acc = pd.DataFrame(data)
    final_micro_f1 = pd.DataFrame(data)
    final_macro_f1 = pd.DataFrame(data)
    final_loss = pd.DataFrame(data)
    final_time = pd.DataFrame(data)
    results = {}
    losses = []

    for i1 in tqdm(range(5), leave = False):
        random.seed(i1)
        np.random.seed(i1)
        torch.manual_seed(i1)
        progress, loss_trial = main()

        final_acc[str(i1)] = progress['final_acc']
        final_micro_f1[str(i1)] = progress['final_micro_f1']
        final_macro_f1[str(i1)] = progress['final_macro_f1']
        final_loss[str(i1)] = progress['final_loss']
        results[str(i1)] = progress['results']
        final_time[str(i1)] = progress['time']
        losses.append(loss_trial)



    final_progress['acc_mean'] = final_acc.mean(axis=1)
    final_progress['acc_std'] = final_acc.std(axis=1)
    final_progress['micro_mean'] = final_micro_f1.mean(axis=1)
    final_progress['micro_std'] = final_micro_f1.std(axis=1)
    final_progress['macro_mean'] = final_macro_f1.mean(axis=1)
    final_progress['macro_std'] = final_macro_f1.std(axis=1)
    final_progress['time_mean'] = final_time.mean(axis=1)
    final_progress['time_std'] = final_time.std(axis=1)


    final_progress['nr_samples'] = progress['nr_samples']
    final_results = pd.DataFrame.from_dict(final_progress)
    final_results.to_csv('/home/pablo/active-learning-pablo/results/test/continual_learning/dark_wo_duplicates2/derpp/a2kb100_2000_new.csv', index=False)
    with open('/home/pablo/active-learning-pablo/results/test/continual_learning/dark_wo_duplicates2/derpp/a2kb100_2000_new.json', 'w') as f:
        json.dump(losses, f)
    #plot_active_process(final_progress)