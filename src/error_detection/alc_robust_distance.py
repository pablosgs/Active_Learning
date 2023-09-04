import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from cleanlab.multilabel_classification.filter import find_label_issues
from cleanlab.count import compute_confident_joint
from src.classification.helpers import split_train_val_test
from src.classification.active_sbert import ClassificationManagerSBERT
from src.alc_utils import corrupt_labels, invert_one_hot, alc_statistics, compute_distance_to_mean, get_rows_with_value, alc_mean_distance, knn_entropy, rank_from_scores, rank_from_scores_k, dled, compute_means
import torch 
import random
import time
import json
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from scipy.spatial import distance

def pass_data(data, indexes):

    [X_lab, X_unlab, X_test, y_lab, y_unlab, y_test, labels] = data
    extracted_X = X_unlab[indexes,:]
    extracted_y = y_unlab[indexes,:]
    X_lab = np.vstack((X_lab, extracted_X))
    y_lab = np.vstack((y_lab, extracted_y))
    X_unlab = np.delete(X_unlab, indexes, axis=0)
    y_unlab = np.delete(y_unlab, indexes, axis=0)
    data = [X_lab, X_unlab, X_test, y_lab, y_unlab, y_test, labels]
    return data

def substitute_rows(array1, array2, row_indexes):
    array1[row_indexes] = array2[row_indexes]
    return array1

def remove_errors(X, y, indexes):

    X = np.delete(X, indexes, axis=0)
    y = np.delete(y, indexes, axis=0)
    return X, y

def store_results(results, accuracy, progress, data, loss):
    micro_f1 = results['micro avg']['f1-score']
    macro_f1 = results['macro avg']['f1-score']
    progress['nr_samples'].append(len(data[0]))
    progress['final_acc'].append(accuracy)
    progress['final_micro_f1'].append(micro_f1)
    progress['final_macro_f1'].append(macro_f1)
    progress['final_loss'].append(loss)
    return progress

def main():

    labels = [
        'Cybercrime',
       'Drugs / Narcotics', 'Financial Crime', 'Goods and Services',
       'Sexual Abuse', 'Violent Crime']
    #df = pd.read_csv('/home/pablo/active-learning-pablo/data/datasets/cflw_final.csv')
    df = pd.read_csv('/home/pablo/active-learning-pablo/data/datasets/cflw_large2_vec.csv')
    X = df.iloc[:, : -len(labels)].values
    y = df.iloc[:, -len(labels) :].values
    corrupted_y, corrupted_samples = corrupt_labels(y, 0.2)

    X_lab, X_unlab, X_test, y_lab, y_unlab, y_test = split_train_val_test(
        X, corrupted_y, train_frac=0.8, val_frac=0.01, test_frac=0.19
    )

    data = [X_lab, X_unlab, X_test, y_lab, y_unlab, y_test, labels]

    progress = {}
    progress['nr_samples'] = []
    progress['final_acc'] = []
    progress['final_micro_f1'] = []
    progress['final_macro_f1'] = []
    progress['final_loss'] = []

    ################ INITIAL PREDICTIONS #####################

    model = ClassificationManagerSBERT(data, mute=True)
    loss = model.fit(n_epochs = 100)
    results, accuracy = model.evaluate_model()
    progress = store_results(results, accuracy, progress, data, loss)


    ############## ERROR DETECTION AND CORRECTION ######################

    #scores = alc_mean_distance(X, corrupted_y)
    means = compute_means(X, corrupted_y)
    predicted_issues = dled(X, corrupted_y, 'euclidean', means)
    #scores = knn_entropy(X, corrupted_y)
    #predicted_issues = rank_from_scores(scores, 80)
    precision, recall, acc, true_positives = alc_statistics(predicted_issues, corrupted_samples, len(y))
    actual_errors = list(set(predicted_issues) & set(corrupted_samples))
    #X, y = remove_errors(X, y, actual_errors)
    #corrupted_y = substitute_rows(corrupted_y, y, actual_errors)
    corrupted_y = y

    ################ SECOND PREDICTIONS #####################

    X_lab, X_unlab, X_test, y_lab, y_unlab, y_test = split_train_val_test(
        X, corrupted_y, train_frac=0.8, val_frac=0.01, test_frac=0.19
    )
    data = [X_lab, X_unlab, X_test, y_lab, y_unlab, y_test, labels]

    model = ClassificationManagerSBERT(data, mute=True)
    loss = model.fit(n_epochs = 100)
    results, accuracy = model.evaluate_model()
    progress = store_results(results, accuracy, progress, data, loss)

    return progress

if __name__ == "__main__":
    
    
    
    data = {str(i): [] for i in range(1)}
    final_progress = {}
    final_acc = pd.DataFrame(data)
    final_micro_f1 = pd.DataFrame(data)
    final_macro_f1 = pd.DataFrame(data)
    final_loss = pd.DataFrame(data)


    for i1 in tqdm(range(3), leave = False):
        random.seed(i1)
        np.random.seed(i1)
        torch.manual_seed(i1)   
        progress = main()
        
        final_acc[str(i1)] = progress['final_acc']
        final_micro_f1[str(i1)] = progress['final_micro_f1']
        final_macro_f1[str(i1)] = progress['final_macro_f1']
        final_loss[str(i1)] = progress['final_loss']


    final_progress['acc_mean'] = final_acc.mean(axis=1)
    final_progress['acc_std'] = final_acc.std(axis=1)
    final_progress['micro_mean'] = final_micro_f1.mean(axis=1)
    final_progress['micro_std'] = final_micro_f1.std(axis=1)
    final_progress['macro_mean'] = final_macro_f1.mean(axis=1)
    final_progress['macro_std'] = final_macro_f1.std(axis=1)


    final_progress['nr_samples'] = progress['nr_samples']
    final_results = pd.DataFrame.from_dict(final_progress)
    final_results.to_csv('/home/pablo/active-learning-pablo/results/test/alc/robustness/gold_02.csv', index=False)
    #plot_active_process(final_progress)