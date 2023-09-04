import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.classification.helpers import split_train_val_test
from src.classification.active_sbert import ClassificationManagerSBERT
from src.classification.plotting import plot_active_process
from actve_learning.query_strategies import random_query, min_confidence, max_score, mean_entropy, max_entropy, label_cardinality
import random

import matplotlib.pyplot as plt
import numpy as np
import scipy

progress = pd.read_csv('results/test/random_query.csv')
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

