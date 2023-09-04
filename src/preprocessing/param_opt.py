import os
import pandas as pd
import numpy as np
import torch
import random
import optuna

from src.preprocessing.text_extraction import TextExtraction
from src.preprocessing.text_preprocessing import TextPreprocessor
from src.vectorization.vectorization_sbert import TextVectorization
from src.classification.helpers import split_train_val_test
from src.classification.active_sbert_opt import fit
from src.classification.plotting import plot_progress


import copy

import numpy as np
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder, normalize



def main():
    
    sampler = optuna.samplers.TPESampler()    
    study = optuna.create_study(sampler=sampler,pruner=optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=5, interval_steps=3),directions=['maximize','maximize'])
    study.optimize(func=fit, n_trials=100)
    
    fig = optuna.visualization.plot_pareto_front(study)
    fig.show()
    fig.savefig('/home/pablo/active-learning-pablo/results/test/pareto.png')

    """"      RAY TUNE
    search_space = {"lr": tune.grid_search([1e-3])}
    #config = {'lr': 1e-3}
    #results = fit(config)
    #print(results)
    tuner = tune.Tuner(fit, param_space=search_space)
    results = tuner.fit()
    dfs = {result.log_dir: result.metrics_dataframe for result in results}
    # Plot by epoch
    ax = None  # This plots everything on the same plot
    for d in dfs.values():
        print(d)
        #ax = d.mean_accuracy.plot(ax=ax, legend=False)
    """

if __name__ == "__main__":
    main()

