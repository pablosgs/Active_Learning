import pandas as pd
import os
import json
import numpy as np
import sys
sys.path.append("../") # go to parent dir


from src.preprocessing.text_extraction import TextExtraction
from src.preprocessing.text_preprocessing import TextPreprocessor
from src.vectorization.vectorization_sbert import TextVectorization
from src.classification.helpers import split_train_val_test
from src.classification.active_sbert import ClassificationManagerSBERT
#from src.classification.plotting import plot_progress

import copy
import time
from tqdm import tqdm

times = {}
for i in tqdm([100, 1000, 2000, 5000, 10000, 20000, 50000, 90000]):
    df = pd.read_csv('/home/pablo/active-learning-pablo/data/datasets/script_data_v2.csv', on_bad_lines='warn')
    df.sum()[-7:]

    df = df.drop(['Unnamed: 0', 'host_id', 'page_id', 'page_version_id',
        'RelatedTagsArray'], axis = 1)

    df = df.head(i)
    start = time.time()

    text_vectorizer = TextVectorization()
    X = text_vectorizer.vectorize(df["html"].values)
    y = df.loc[:-1].values

    #df_train = pd.DataFrame(np.concatenate((X, y), axis=1))
    end = time.time()

    elapsed = end - start
    times[str(i)] = elapsed

final_times = pd.DataFrame.from_dict(times)
with open('/home/pablo/active-learning-pablo/results/test/active_learning/dark_wo_duplicates/sampling_size/embed_time.json', 'w') as f:
    json.dump(final_times, f)

