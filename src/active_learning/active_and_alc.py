import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import sys
sys.path.insert(1, '/home/pablo/active-learning-pablo')
from src.classification.helpers import split_train_val_test
from src.classification.active_sbert import ClassificationManagerSBERT
from cleanlab.multilabel_classification.filter import find_label_issues
from cleanlab.count import compute_confident_joint
#from src.classification.plotting import plot_active_process
from active_learning.query_strategies import random_query, random_order, min_confidence, max_score, mean_entropy, max_entropy, label_cardinality, mean_max_loss, badge, score_label_card, expected_gradient, DAL, cvirs, alucs, dalucs, clue
import torch 
import random
import time
import json
from src.error_detection.alc_utils import retag, invert_one_hot, alc_statistics, knn_entropy, rank_from_scores

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



    # Create dataset and initialize everything
    #create_ganymede_dataset(
    #    fw=fw, labels=abuse_labels, overwrite_dataset=overwrite_dataset, deduplication_method="title", n_folders=10)

    #X, y, labels = create_embeddings(fw, abuse_labels, model_name, restart_pipeline, overwrite_model, subset_size)
    """"
    df_train = pd.read_csv('/home/pablo/active-learning-pablo/data/datasets/toxic_train_vec_final.csv')
    df_test = pd.read_csv('/home/pablo/active-learning-pablo/data/datasets/toxic_test_vec_final.csv')
    df_val = pd.read_csv('/home/pablo/active-learning-pablo/data/datasets/toxic_val_vec_final.csv')
    labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate', 'non_toxic']
    val_toxic = df_val[df_val[df_val.columns.values[-1]] == 0]

    df_train = pd.concat([val_toxic, df_train])
    df_lab = df_train.copy()
    df_unlab = df_train.copy()
    df_unlab = df_unlab[:-500] #df_unlab.drop(df_unlab.tail(500).index,inplace=True)
    df_lab = df_lab[-500:]#df_lab.drop(df_lab.head(df_lab.shape[0]-500))
    y_unlab = df_unlab[df_unlab.columns.values[384:]].values
    X_unlab = df_unlab[df_unlab.columns.values[:384]].values
    y_test = df_test[df_test.columns.values[384:]].values
    X_test = df_test[df_test.columns.values[:384]].values
    y_lab = df_lab[df_lab.columns.values[384:]].values
    X_lab = df_lab[df_lab.columns.values[:384]].values
    """

    labels = [
        'Cybercrime',
       'Drugs / Narcotics', 'Financial Crime', 'Goods and Services',
       'Sexual Abuse', 'Violent Crime']
    df = pd.read_csv('/home/pablo/active-learning-pablo/data/datasets/cflw_large2_vec.csv')
    # financial_df = df[df['386'] == 1]
    # non_financial_df = df[df['386'] == 0]
    # cyber_df = df[df['384'] == 1]
    # drugs_df = df[df['385'] == 1]
    # goods_df = df[df['387'] == 1]
    # abuse_df = df[df['388'] == 1]
    # violent_df = df[df['389'] == 1]
    #df = pd.concat([non_financial_df,financial_df.sample(10_000)])
    #df = pd.concat([financial_df.sample(1160),cyber_df.sample(1160) , drugs_df.sample(1160) , goods_df.sample(1160) , abuse_df.sample(1160) , violent_df.sample(1160)])
    y = df.iloc[:, -len(labels) :].values
    X = df.iloc[:, : -len(labels)].values

    X_init = X[:6000]
    y_init = y[:6000]
    X_unlab = X[6000:]
    y_unlab = y[6000:]
    X_test = X[6000:]
    y_test = y[6000:]

    data = [X_init, X_unlab, X_test, y_init, y_unlab, y_test, labels]
    #X_unlab, X_lab, X_test, y_unlab, y_lab, y_test = split_train_val_test(
    #    X, y, train_frac=0.5, val_frac=0.41, test_frac=0.09
    #)
    #data = [X_lab, X_unlab, X_test, y_lab, y_unlab, y_test, labels]

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

    timer = []
    beta = 1
    final_results = {}
    precisions = []
    recalls = []
    accs = []
    trues = []
    actual_errors = []
    iterations = []

    for i in tqdm(range(1), leave = True): #Add stopping criteria

        #1. Train model
        model = ClassificationManagerSBERT(data, mute=True)
        loss = model.fit(n_epochs = 100)

        #2. Evaluate model and save results
        results, accuracy = model.evaluate_model()
        progress = store_results(results, accuracy, progress, data, loss, 0)
        final_results[str(i)] = results
    

        #3. Make predictions on U
        pred = model.predict_unlab()

        #4. Query

        batch_size = 1000
        indexes = mean_max_loss(pred, batch_size)
        

        # 5. Label with model
        preds_selected = pred[indexes,:]
        predicted_labels = (preds_selected > 0.5).astype(int)
        labels_list = invert_one_hot(predicted_labels)
        actual_labels = data[4][indexes,:]
        selected_X = data[1][indexes,:]

        #predicted_issues = find_label_issues(labels=labels_list,pred_probs=preds_selected,return_indices_ranked_by="self_confidence", filter_by = 'both', frac_noise=1)
        
        avg_train_loss, confidence = model.fit(n_epochs = 100, datamap = 1)
        scores = datamap_conf(confidence)
        #avg_train_loss, scores, mapping = model.fit(n_epochs = 100, curriculum = 1)
        #predicted_issues = np.where(mapping)[0]
        predicted_issues = rank_from_scores(scores, percentile)
        with open('/home/pablo/active-learning-pablo/results/test/alc/real_preds/trial.txt', 'w') as fp:
            for item in predicted_issues:
                # write each item on a new line
                fp.write("%s\n" % str(item))
        print('Done')
        
        
        comparison = np.equal(actual_labels, predicted_labels)
        row_sums = np.sum(comparison, axis=1)
        mask = row_sums != 6
        actual_issues = np.nonzero(mask == 1)[0]

        precision, recall, accuracy, true_positives = alc_statistics(predicted_issues, actual_issues, len(indexes))

        precisions.append(precision)
        recalls.append(recall)
        accs.append(accuracy)
        trues.append(len(predicted_issues))
        actual_errors.append(len(actual_issues))
        iterations.append(i)


        #5. Pass sampled instances
        data = pass_data(data, indexes)


    return precisions, recalls, accs, trues, actual_errors, iterations


if __name__ == "__main__":
    
    
    data = {str(i): [] for i in range(1)}
    final_progress = {}
    final_precision = pd.DataFrame(data)
    final_recall = pd.DataFrame(data)
    final_accuracy = pd.DataFrame(data)
    final_trues = pd.DataFrame(data)
    final_actual = pd.DataFrame(data)
    #final_time = pd.DataFrame(data)
    results = {}

    for i1 in tqdm(range(1), leave = False):
        random.seed(i1)
        np.random.seed(i1)
        torch.manual_seed(i1)   
        precision, recall, accuracy, predicted_errors, actual_errors, percentiles = main()
        
        #numbers_df = pd.DataFrame(numbers, columns = ['Cybercrime','Drugs / Narcotics', 'Financial Crime', 'Goods and Services','Sexual Abuse', 'Violent Crime'])
        #numbers_df.to_csv('/home/pablo/active-learning-pablo/results/test/active_learning/dark_wo_duplicates/badge_distribution.csv', index=False)

        #print('egl: ', timer)

        final_precision[str(i1)] = precision
        final_recall[str(i1)] = recall
        final_accuracy[str(i1)] = accuracy
        final_trues[str(i1)] = predicted_errors
        final_actual[str(i1)] = actual_errors
        #final_time[str(i1)] = progress['time']
        #results[str(i1)] = result

    final_progress['percentiles'] = np.asarray(percentiles)
    final_progress['precision_mean'] = final_precision.mean(axis=1)
    final_progress['precision_std'] = final_precision.std(axis=1)
    final_progress['recall_mean'] = final_recall.mean(axis=1)
    final_progress['recall_std'] = final_recall.std(axis=1)
    final_progress['acc_mean'] = final_accuracy.mean(axis=1)
    final_progress['acc_std'] = final_accuracy.std(axis=1)
    final_progress['predicted_mean'] = final_trues.mean(axis=1)
    final_progress['predicted_std'] = final_trues.std(axis=1)
    final_progress['actual_mean'] = final_actual.mean(axis=1)
    final_progress['actual_std'] = final_actual.std(axis=1)

    #final_progress['nr_samples'] = progress['nr_samples']
    final_results = pd.DataFrame.from_dict(final_progress)
    final_results.to_csv('/home/pablo/active-learning-pablo/results/test/alc/al_and_alc/knn_entropy.csv', index=False)
    #with open('/home/pablo/active-learning-pablo/results/test/alc/datamap.json', 'w') as f:
        #json.dump(results, f)
    #plot_active_process(final_progress)