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
from sklearn.model_selection import train_test_split
import json 
import time
from copy import deepcopy

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
    progress['nr_samples'].append(len(data[0]))
    progress['final_acc'].append(accuracy)
    progress['final_micro_f1'].append(micro_f1)
    progress['final_macro_f1'].append(macro_f1)
    progress['final_loss'].append(loss)
    return progress

def shrink_perturb(model, lamda=0.5, sigma=0.01):
    for (name, param) in model.named_parameters():
        if 'weight' in name:   # just weights
            nc = param.shape[0]  # cols
            nr = param.shape[1]  # rows
            for i in range(nr):
                for j in range(nc):
                    param.data[j][i] = (lamda * param.data[j][i]) + torch.normal(0.0, sigma, size=(1,1))
    return

def shrink_perturb2(model, shrink, perturb):
        # using a randomly-initialized model as a noise source respects how different kinds 
        # of parameters are often initialized differently
        input_layer_size = 384
        hidden_layer_size = 100
        output_layer_size = 6
        dropout = 0.5
        #new_init = SBERTModel(input_layer_size, hidden_layer_size, output_layer_size, self.device, dropout)
        #params1 = new_init.parameters()
        #params2 = model.parameters()
        #for p1, p2 in zip(*[params1, params2]):
        #    p1.data = deepcopy(shrink * p2.data + perturb * p1.data)
        #return new_init
        for p1 in model.parameters():
            p1.data = deepcopy(shrink * p1.data + torch.normal(0, perturb, size = p1.data.shape))

        return

def main():
    experiment_name = "test"
    dataset_name = "ganymede_data_mini"
    model_name = "test"
    overwrite_dataset = False  # Allow dataset to be overwritten
    overwrite_results = False  # Allow experiment results to be overwritten
    overwrite_model = True  # Allow model results to be overwritten
    restart_pipeline = False  # Force execution of all steps in the pipeline
    subset_size = 0.05  # Fraction of total samples to be used

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
    label_order = [
        'Financial Crime', 'Drugs / Narcotics', 'Cybercrime', 'Goods and Services',
       'Sexual Abuse', 'Violent Crime']
    divided_df = {}
    df = pd.read_csv('/home/pablo/active-learning-pablo/data/datasets/cflw_large2_vec.csv')
    #financial_df = df[df['386'] == 1]
    #non_financial_df = df[df['386'] == 0]
    #
    #df = pd.concat([non_financial_df,financial_df.sample(10_000)])

    train, test = train_test_split(df, test_size=0.19)

    y_test = test.iloc[:, -len(labels) :].values
    X_test = test.iloc[:, : -len(labels)].values

    divided_df['Cybercrime'] = train.loc[(df['384'] == 1)]
    divided_df['Drugs / Narcotics'] = train.loc[(df['385'] == 1)]
    divided_df['Financial Crime'] = train.loc[(df['386'] == 1)]
    divided_df['Goods and Services'] = train.loc[(df['387'] == 1)]
    divided_df['Sexual Abuse'] = train.loc[(df['388'] == 1)]
    divided_df['Violent Crime'] = train.loc[(df['389'] == 1)]
    

    exp_df = pd.concat([divided_df['Drugs / Narcotics'], divided_df['Financial Crime'], divided_df['Cybercrime'], divided_df['Goods and Services'], divided_df['Violent Crime']])

    lab, unlab = train_test_split(exp_df, test_size=0.5)

    unlab = pd.concat([unlab, divided_df['Sexual Abuse']])

    y_lab = lab.iloc[:, -len(labels) :].values
    X_lab = lab.iloc[:, : -len(labels)].values

    y_unlab = unlab.iloc[:, -len(labels) :].values
    X_unlab = unlab.iloc[:, : -len(labels)].values


    data = [X_lab, X_unlab, X_test, y_lab, y_unlab, y_test, labels]

    progress = {}
    progress['nr_samples'] = []
    progress['final_acc'] = []
    progress['final_micro_f1'] = []
    progress['final_macro_f1'] = []
    progress['final_loss'] = []

    final_results = {}
    count_results = 0

    beta = 1
    shrink = 1
    perturb = 0.1
    timer = []
    overall_loss = []
    buffer_data = {}
    buffer_data['X'] = np.zeros(384)
    buffer_data['labels'] = np.zeros(6)
    buffer_data['logits'] = np.zeros(6)

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

    for i in tqdm(range(5), leave = True, position = 0): #Add stopping criteria
        #if i == 0:
        #    numbers = data[3].sum(axis=0)
        #elif i == 19:
        #    numbers = np.vstack((numbers, data[3].sum(axis=0)))
        
        #1. Train model
        model = ClassificationManagerSBERT(data, mute=True)
        #if i >= 1:
            #teacher = ClassificationManagerSBERT(data, mute=True)
            #teacher.model.load_state_dict(torch.load('/home/pablo/active-learning-pablo/results/test/continual_learning/models/model.pth'))
            #model.model.load_state_dict(torch.load('/home/pablo/active-learning-pablo/results/test/continual_learning/models/model.pth'))
            #shrink_perturb(model.model, shrink, perturb)
            #loss = model.fit(teacher = teacher, n_epochs = 100, dis = 0, ewc = 0, derpp = 1, buffer = buffer_data)
        #else:
        loss = model.fit(n_epochs = 100, dis = 0, ewc = 0)
        #pred_lab = np.zeros(data[3].shape)
        #index_buffer = random_query(pred_lab, int(0.1*pred_lab.shape[0]))
        #buffer_data = extend_buffer(buffer_data, data, index_buffer, pred_lab)
        overall_loss.append(loss)
        torch.save(model.model.state_dict(), '/home/pablo/active-learning-pablo/results/test/continual_learning/models/model.pth')

        #2. Evaluate model and save results
        results, accuracy = model.evaluate_model()
        progress = store_results(results, accuracy, progress, data, loss)
        final_results[str(count_results)] = results
        count_results += 1
    

        #3. Make predictions on U
        pred = model.predict_unlab()
        #embedder = model.model.get_act() 
        embeddings = model.produce_embeddings()

        #4. Query
        beta = 2^(-i)
        batch_size = 1000
        #start = time.time()
        #indexes = score_label_card(pred, 500, beta)
        #indexes = badge(pred, embeddings, batch_size)
        #indexes = expected_gradient(pred, embeddings, batch_size)
        #indexes = DAL(data, batch_size)
        #indexes = dalucs(pred, data, batch_size, beta)
        indexes = mean_max_loss(pred, batch_size)
        #indexes = label_cardinality(pred, 500)
        #indexes = random_query(pred, batch_size)
        #indexes = clue(embeddings, pred, batch_size)
        #assert not all(i >= X_unlab.shape[0] for i in indexes)
        #Add something to check if repeated numbers
        #end = time.time()
        #timer.append(str(end-start))
        #5. Pass sampled instances
        data = pass_data(data, indexes)
        """"
        if i == 4 and label == 'Violent Crime':
            predictions_test = pd.DataFrame(y_pred, columns = ['Cybercrime','Drugs / Narcotics', 'Financial Crime', 'Goods and Services','Sexual Abuse', 'Violent Crime'])
            predictions_test.to_csv('results/test/dark/DA/analysis/badge_pred_test.csv', index=False)

            true_test = pd.DataFrame(y_true, columns = ['Cybercrime','Drugs / Narcotics', 'Financial Crime', 'Goods and Services','Sexual Abuse', 'Violent Crime'])
            true_test.to_csv('results/test/dark/DA/analysis/badge_true_test.csv', index=False)
        """

    #plot_active_process(nr_samples, final_acc, final_f1, final_loss)
    #plot_active_process(progress)
    #print(pred)
    #print('Final accuracy: ', results['samples avg']['precision'])

    return progress, final_results, overall_loss

if __name__ == "__main__":
    
    
    
    data = {str(i): [] for i in range(5)}
    final_progress = {}
    final_acc = pd.DataFrame(data)
    final_micro_f1 = pd.DataFrame(data)
    final_macro_f1 = pd.DataFrame(data)
    final_loss = pd.DataFrame(data)
    results = {}
    losses = []

    for i1 in tqdm(range(5), leave = True, position = 0):
        random.seed(i1)
        np.random.seed(i1)
        torch.manual_seed(i1)
        #progress, result, numbers, timer = main()
        progress, result, loss_trial = main()
        
        #numbers_df = pd.DataFrame(numbers, columns = ['Cybercrime','Drugs / Narcotics', 'Financial Crime', 'Goods and Services','Sexual Abuse', 'Violent Crime'])
        #numbers_df.to_csv('results/test/dark/DA/analysis/badge_distribution.csv', index=False)

        #print(timer)

        final_acc[str(i1)] = progress['final_acc']
        final_micro_f1[str(i1)] = progress['final_micro_f1']
        final_macro_f1[str(i1)] = progress['final_macro_f1']
        final_loss[str(i1)] = progress['final_loss']
        results[str(i1)] = result
        losses.append(loss_trial)


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
    final_results.to_csv('/home/pablo/active-learning-pablo/results/test/continual_learning/dark_wo_duplicates2/DA3/retrain_abuse.csv', index=False)
    with open('/home/pablo/active-learning-pablo/results/test/continual_learning/dark_wo_duplicates2/DA3/retrain_abuse.json', 'w') as f:
        json.dump(results, f)
    #plot_active_process(final_progress)