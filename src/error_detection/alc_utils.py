import pandas as pd
import math
import numpy as np
import random
from scipy.spatial import distance
from nessie.detectors.knn_entropy import KnnErrorDetector
from cleanlab.internal.multilabel_utils import int2onehot, onehot2int
import torch
from scipy.stats import median_abs_deviation
from scipy.spatial.distance import cdist


def corrupt_labels(labels, p):
    num_flips = int(labels.shape[0] * p)

    # Randomly choose rows to flip
    flip_indices = np.random.choice(labels.shape[0], size=num_flips, replace=False)

    # Flip one value in each selected row
    flipped_matrix = np.copy(labels)
    flipped_matrix[flip_indices, np.random.randint(labels.shape[1], size=num_flips)] = np.logical_not(flipped_matrix[flip_indices, np.random.randint(labels.shape[1], size=num_flips)])

    return flipped_matrix, flip_indices


def invert_one_hot(array):
    result = []
    for row in array:
        indexes = [i for i, value in enumerate(row) if value]
        result.append(indexes)
    return result

def convert_boolean_to_list(array):
    row_indices, column_indices = np.where(array)
    _, row_counts = np.unique(row_indices, return_counts=True)
    result = np.split(column_indices, np.cumsum(row_counts)[:-1])
    result = [row.tolist() for row in result]
    return result

def alc_statistics(predictions, solutions, total_number):
    true_positive = len(set(predictions) & set(solutions))
    false_positive = len(set(predictions) - set(solutions))
    false_negative = len(set(solutions) - set(predictions))
    true_negative = len(set(list(range(total_number + 1))) - (set(predictions) | set(solutions)))

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative) if (true_positive + true_negative + false_positive + false_negative) > 0 else 0

    return precision, recall, accuracy, np.asarray(list(set(predictions) & set(solutions)))


def rank_from_scores(scores, percentile):
     return np.argwhere(scores > np.percentile(scores, percentile)).flatten()

def rank_from_scores_k(scores, k):
     return np.argwhere(scores > k * median_abs_deviation(scores)).flatten()

def retrieve_batch_by_scores(scores,batch_size):
    return np.argpartition(-scores, batch_size-1, axis=0)[:batch_size]

def retrieve_by_threshold(scores, threshold):
    return np.argwhere(scores > threshold).flatten()

def mad(arr):
    """ Median Absolute Deviation: a "Robust" version of standard deviation.
        Indices variabililty of the sample.
        https://en.wikipedia.org/wiki/Median_absolute_deviation 
    """
    arr = np.ma.array(arr).compressed() # should be faster to not use masked arrays.
    med = np.median(arr)
    return np.median(np.abs(arr - med))


def compute_distance_to_mean(vectors: np.ndarray, labels: np.ndarray, mean_vectors, f) -> np.ndarray:
    n = len(vectors)
    distances = np.zeros(n)
    for i in range(n):
        vector = vectors[i]
        positive_labels = np.where(labels[i])[0]
        for j in positive_labels:
            dists_sample = []
            mean_vector = mean_vectors[str(j)]
            dists_sample.append(f(vector, mean_vector))

        distances[i] = max(dists_sample)

    return distances

def get_rows_with_value(arr, column_index):
    column_values = arr[:, column_index]
    row_indexes = np.where(column_values == 1)[0]
    return row_indexes

def alc_mean_distance(X, corrupted_y):
    means = {}
    for i in range(6):
        means[str(i)] = np.mean(X[get_rows_with_value(corrupted_y, i),:],axis = 0)
    distances = compute_distance_to_mean(X, corrupted_y, means, distance.cosine) #distance.cosine
    return distances

def knn_entropy(X, corrupted_y):
    labels = invert_one_hot(corrupted_y)
    detector = KnnErrorDetector(k = 100)
    scores = detector.score(labels, X)
    #idx = np.argwhere(scores).flatten()
    return scores

######### DATAMAP CONFIDENCE #########
def select_probabilities(one_hot_array, probabilities):
    mult = torch.mul(one_hot_array, probabilities)
    nonzero_values = mult.clone()
    nonzero_values[mult == 0] = float('inf')
    return torch.min(nonzero_values, dim=1).values

def datamap_conf(confidence):
    df = pd.DataFrame.from_dict(confidence)
    uncertainty = 1 - df
    scores = uncertainty.sum(axis=1).to_numpy()
    return scores
#######################################

######## Curriculum

def compute_lambda(y_gold: np.ndarray, y_pred: np.ndarray, losses: np.ndarray) -> float:
    # lambda is the average loss of correctly classified instances
    correct_indices = y_gold == y_pred
    losses_for_correct_instances = losses[np.argwhere(correct_indices.sum(axis = 1) == 6).flatten()]#losses[correct_indices]
    l = losses_for_correct_instances.mean()
    return float(l)

def _sample_easy(lambda_: float, losses: np.ndarray) -> np.ndarray:
        # Easy instances are instances with have a  loss of `l` or less
        # Return a boolean array saying whether an instance belongs to the new easy dataset

        easy_mask = losses <= lambda_
        return easy_mask

def _sample_hard(lambda_: float, delta: float, losses: np.ndarray) -> np.ndarray:
    # Hard instances are instances with have a  loss of greater than `l`
    # we sample the `delta` easiest percent of the hardest

    # Return a boolean array saying whether an instance belongs to the new hard dataset
    assert 0.0 <= delta <= 1.0

    n = len(losses)

    k = int(n * delta)
    assert 0 <= k <= n

    hard_mask = losses > lambda_

    hard_indices = []
    loss_indices_sorted = np.argsort(losses)

    # Go through the indices from lowest to highest loss and collect the first `k` hard instances
    for i in loss_indices_sorted:
        if hard_mask[i]:
            hard_indices.append(i)

        if len(hard_indices) >= k:
            break

    result_mask = np.zeros(n, dtype=bool)
    result_mask[hard_indices] = True

    return result_mask

def _update_stat(hard_mask: np.ndarray, losses: np.ndarray):
        num_of_hard_instances = np.count_nonzero(hard_mask)
        new_scores = hard_mask * (losses + 1 / num_of_hard_instances)
        return new_scores

def update_mapping(new_dataset_mask: np.ndarray):
        # ``new_dataset`` is array of bool saying whether that instance is part of the new dataset

        # We build a mapping that maps instances from [0, |new_dataset|]
        # to indices in the original dataset
        count = 0
        mapping = {}
        for idx, e in enumerate(new_dataset_mask):
            # Instance is part of new dataset
            if e:
                mapping[count] = idx

                count += 1

        return mapping


############# LEITNER ####################

def new_score_leitner(queues, losses: np.ndarray, scores):
        num_of_q0_instances = len(queues[0])

        for idx in queues[0]:
            scores[idx] += losses[idx] + 1 / num_of_q0_instances

        return scores


def build_training_mask_leitner(training_mask, epoch: int, queues) -> np.ndarray:
    new_mask = np.zeros_like(training_mask)

    for q, queue in enumerate(queues):
        if epoch % (2**q) != 0:
            # The time for this queue has not come yet
            continue

        for idx in queue:
            new_mask[idx] = True

    return new_mask

def compute_new_queues(y_gold: np.ndarray, y_pred: np.ndarray, queues, training_mask):
        assert len(y_gold) == len(y_pred)

        correct_indices = y_gold == y_pred

        new_queues = [[] for _ in range(10)]

        for q, queue in enumerate(queues):
            for idx in queue:
                # If the instance was not part of the training this epoch, we just skip
                if not training_mask[idx]:
                    continue

                # If the item was correctly classified, we promote it,
                # otherwise, we demote it to queue 0
                if sum(correct_indices[idx]) == 6:
                    new_queue = min(10, q + 1)
                else:
                    new_queue = 0

                new_queues[new_queue].append(idx)

        return new_queues
##############
def max_loss(preds, class_pred):
    #class_pred = (preds > 0.5).astype(float)
    c_weights = torch.from_numpy((len(class_pred)-np.sum(class_pred,axis=0))/np.sum(class_pred,axis=0))
    loss = torch.nn.BCELoss(reduction="none", weight=c_weights)
    losses = loss(torch.from_numpy(preds), torch.from_numpy(class_pred))
    mean_losses = torch.max(losses, dim = 1).values.detach().cpu().numpy()
    return mean_losses

def retag(predictions, labels):
    class_pred = (predictions > 0.5).astype(float)
    comparison = np.equal(class_pred, labels)
    row_sums = np.sum(comparison, axis=1)
    mask = row_sums != 6
    idx = np.nonzero(mask == 1)[0]
    return idx

def retag_plus(predictions, labels, condition_value):
    class_pred = (predictions > 0.5).astype(float)
    comparison = np.equal(class_pred, labels)
    condition_mask = np.logical_and(comparison == 0, predictions < condition_value)
    idx = np.where(np.any(condition_mask, axis=1))[0]
    return idx

def dled(X, labels, metric, means):
    #means = compute_means(X, labels)
    distance_matrix = cdist(X, means,metric)
    col_indices = invert_one_hot(labels)
    min_positions = np.argmin(distance_matrix, axis=1)
    result = [bool(np.isin(minimum, cols)) for minimum, cols in zip(min_positions, col_indices)]
    idx = np.argwhere(np.asarray(result) == 0).flatten()
    for index in idx:
        if abs(distance_matrix[index][min_positions[index]] - distance_matrix[index][col_indices[0]]) < 0.2:
            idx = np.delete(idx, np.where(idx == index))
    return idx

    
def compute_means(X, labels):
    for i in range(6):
        if i == 0:
            means = np.mean(X[get_rows_with_value(labels, i),:],axis = 0)
        else:
            means = np.vstack([means, np.mean(X[get_rows_with_value(labels, i),:],axis = 0)])
    return means





