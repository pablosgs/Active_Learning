import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.classification.helpers import split_train_val_test
from src.classification.active_sbert import ClassificationManagerSBERT
from src.classification.plotting import plot_active_process
from actve_learning.query_strategies import random_query, min_confidence, max_score, mean_entropy, max_entropy, label_cardinality, mean_max_loss, badge, score_label_card, expected_gradient, DAL, cvirs, alucs, dalucs
import torch 
import random

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

def store_results(results, accuracy, progress, data, loss):
    micro_f1 = results['micro avg']['f1-score']
    macro_f1 = results['macro avg']['f1-score']
    progress['nr_samples'].append(data[0].shape[0] - 141)
    progress['final_acc'].append(accuracy)
    progress['final_micro_f1'].append(micro_f1)
    progress['final_macro_f1'].append(macro_f1)
    progress['final_loss'].append(loss)
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
    df = pd.read_csv('/home/pablo/active-learning-pablo/data/datasets/script_data_v2_vec.csv')
    financial_df = df[df['386'] == 1]
    non_financial_df = df[df['386'] == 0]
    df = pd.concat([non_financial_df,financial_df.sample(15_000)])
    y = df.iloc[:, -len(labels) :].values
    X = df.iloc[:, : -len(labels)].values

    X_unlab, X_lab, X_test, y_unlab, y_lab, y_test = split_train_val_test(
        X, y, train_frac=0.7, val_frac=0.013, test_frac=0.287
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

    beta = 1
    for i in tqdm(range(15), leave = True): #Add stopping criteria
        
        #1. Train model
        model = ClassificationManagerSBERT(data, mute=True)
        loss = model.fit(n_epochs = 70)

        #2. Evaluate model and save results
        results, accuracy = model.evaluate_model()
        progress = store_results(results, accuracy, progress, data, loss)
    

        #3. Make predictions on U
        pred = model.predict_unlab()
        #embedder = model.model.get_act() 
        embeddings = model.produce_embeddings()

        #4. Query
        beta = 2^(-i)
        #indexes = score_label_card(pred, 500, beta)
        #indexes = badge(pred, embeddings, 1000)
        #indexes = DAL(data, 500)
        #indexes = dalucs(pred, data, 500, beta)
        #indexes = label_cardinality(pred, 500)
        indexes = random_query(pred, 500)
        #assert not all(i >= X_unlab.shape[0] for i in indexes)
        #Add something to check if repeated numbers

        #5. Pass sampled instances
        data = pass_data(data, indexes)

    #plot_active_process(nr_samples, final_acc, final_f1, final_loss)
    #plot_active_process(progress)
    #print(pred)
    #print('Final accuracy: ', results['samples avg']['precision'])

    return progress

if __name__ == "__main__":
    
    
    data = {str(i): [] for i in range(10)}
    final_progress = {}
    final_acc = pd.DataFrame(data)
    final_micro_f1 = pd.DataFrame(data)
    final_macro_f1 = pd.DataFrame(data)
    final_loss = pd.DataFrame(data)

    for i1 in tqdm(range(10), leave = False):
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
    final_progress['loss_mean'] = final_loss.mean(axis=1)
    final_progress['loss_std'] = final_loss.std(axis=1)

    final_progress['nr_samples'] = progress['nr_samples']
    final_results = pd.DataFrame.from_dict(final_progress)
    final_results.to_csv('results/test/dark/dalucs.csv', index=False)
    #plot_active_process(final_progress)
