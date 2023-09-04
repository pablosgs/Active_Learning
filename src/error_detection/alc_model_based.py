import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from cleanlab.multilabel_classification.filter import find_label_issues
from cleanlab.count import compute_confident_joint
from src.classification.helpers import split_train_val_test
from src.classification.active_sbert import ClassificationManagerSBERT
from src.alc_utils import corrupt_labels, invert_one_hot, alc_statistics, max_loss, rank_from_scores, retag, retag_plus
import torch 
import random
import time
import json
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from nessie.detectors import Retag

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

def store_results(results, accuracy, progress, data, loss, elapsed):
    micro_f1 = results['micro avg']['f1-score']
    macro_f1 = results['macro avg']['f1-score']
    progress['nr_samples'].append(len(data[0]))
    progress['final_acc'].append(accuracy)
    progress['final_micro_f1'].append(micro_f1)
    progress['final_macro_f1'].append(macro_f1)
    progress['final_loss'].append(loss)
    progress['time'].append(elapsed)
    return progress

def main():
    experiment_name = "test"
    dataset_name = "ganymede_data_mini"
    model_name = "test"
    overwrite_dataset = False  # Allow dataset to be overwritten
    overwrite_results = False  # Allow experiment results to be overwritten
    overwrite_model = True  # Allow model results to be overwritten
    restart_pipeline = False  # Force execution of all steps in the pipeline
    subset_size = 0.05  # Fraction of total samples to be used


    labels = [
        'Cybercrime',
       'Drugs / Narcotics', 'Financial Crime', 'Goods and Services',
       'Sexual Abuse', 'Violent Crime']
    df = pd.read_csv('/home/pablo/active-learning-pablo/data/datasets/cflw_final.csv')

    kf = KFold(n_splits = 5, shuffle = True, random_state = 2)
    splits = list(kf.split(df))
    X = df.iloc[:, : -len(labels)].values
    y = df.iloc[:, -len(labels) :].values
    corrupted_y, corrupted_samples = corrupt_labels(y, 0.1)
    print(len(corrupted_samples))
    predicted_issues = []
    precisions = []
    recalls = []
    accs = []
    trues = []
    percentiles = [1] #[85, 90, 95, 99, 99.5, 99.9]

    for percentile in percentiles:
        start = time.time()
        for split_indexes in splits:
            X_train = X[split_indexes[0].tolist(), :]     
            y_train = corrupted_y[split_indexes[0].tolist(), :] 
            X_test = X[split_indexes[1].tolist(), :]     
            y_test = corrupted_y[split_indexes[1].tolist(), :]
            X_val =  np.copy(X_test)         
            y_val =  np.copy(y_test)  
        

            data = [X_train, X_val, X_test, y_train, y_val, y_test, labels]

            
            model = ClassificationManagerSBERT(data, mute=True)

            loss = model.fit(n_epochs = 100)
            preds = model.predict_unlab()

            y_val_list = invert_one_hot(y_val)

            #scores = max_loss(preds, y_val)
            #ranked_label_issues = rank_from_scores(scores, percentile)
            #predicted_issues.extend()
            #ranked_label_issues = find_label_issues(labels=y_val_list,pred_probs=preds,return_indices_ranked_by="self_confidence", filter_by = 'both', frac_noise=1)
            #ranked_label_issues = ranked_label_issues[200:-1 or None]
        
            ranked_label_issues = retag(preds, y_val)
            #ranked_label_issues = np.intersect1d(ranked_label_issues1, ranked_label_issues2)
            #ranked_label_issues = retag_plus(preds, y_val, 0.7)
            predicted_issues.extend(np.array(split_indexes[1])[ranked_label_issues].tolist())

            #results, accuracy = model.evaluate_model()
            #progress = store_results(results, accuracy, progress, data, loss, 0)
        end = time.time()
        elapsed = end-start
        print(elapsed)
        precision, recall, accuracy, true_positives = alc_statistics(predicted_issues, corrupted_samples, len(y))
        precisions.append(precision)
        recalls.append(recall)
        accs.append(accuracy)
        trues.append(len(predicted_issues))
        #y_val_list = invert_one_hot(y_val)
    return precisions, recalls, accs, trues, percentiles

if __name__ == "__main__":
    
    
    data = {str(i): [] for i in range(1)}
    final_progress = {}
    final_precision = pd.DataFrame(data)
    final_recall = pd.DataFrame(data)
    final_accuracy = pd.DataFrame(data)
    final_trues = pd.DataFrame(data)
    #final_time = pd.DataFrame(data)
    results = {}

    for i1 in tqdm(range(1), leave = False):
        random.seed(i1)
        np.random.seed(i1)
        torch.manual_seed(i1)   
        precision, recall, accuracy, true_positives, percentiles = main()
        
        #numbers_df = pd.DataFrame(numbers, columns = ['Cybercrime','Drugs / Narcotics', 'Financial Crime', 'Goods and Services','Sexual Abuse', 'Violent Crime'])
        #numbers_df.to_csv('/home/pablo/active-learning-pablo/results/test/active_learning/dark_wo_duplicates/badge_distribution.csv', index=False)

        #print('egl: ', timer)

        final_precision[str(i1)] = precision
        final_recall[str(i1)] = recall
        final_accuracy[str(i1)] = accuracy
        final_trues[str(i1)] = true_positives
        #final_time[str(i1)] = progress['time']
        #results[str(i1)] = result

    final_progress['percentiles'] = np.asarray(percentiles)
    final_progress['precision_mean'] = final_precision.mean(axis=1)
    final_progress['precision_std'] = final_precision.std(axis=1)
    final_progress['recall_mean'] = final_recall.mean(axis=1)
    final_progress['recall_std'] = final_recall.std(axis=1)
    final_progress['acc_mean'] = final_accuracy.mean(axis=1)
    final_progress['acc_std'] = final_accuracy.std(axis=1)
    final_progress['trues_mean'] = final_trues.mean(axis=1)
    final_progress['trues_std'] = final_trues.std(axis=1)

    #final_progress['nr_samples'] = progress['nr_samples']
    final_results = pd.DataFrame.from_dict(final_progress)
    #final_results.to_csv('/home/pablo/active-learning-pablo/results/test/alc/noise_01/retag2.csv', index=False)
    #with open('/home/pablo/active-learning-pablo/results/test/alc/datamap.json', 'w') as f:
        #json.dump(results, f)
    #plot_active_process(final_progress)