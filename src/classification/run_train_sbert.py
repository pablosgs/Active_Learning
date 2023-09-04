import os
import pandas as pd
import numpy as np

from src.data.ganymede import create_ganymede_dataset, load_html
from src.data.setup_experiment import FileWarehouse
from src.preprocessing.text_extraction import TextExtraction
from src.preprocessing.text_preprocessing import TextPreprocessor
from src.vectorization.vectorization_sbert import TextVectorization
from src.classification.helpers import split_train_val_test
from src.classification.classification_sbert import ClassificationManagerSBERT


def train_sbert(
    fw: FileWarehouse,
    labels: list,
    model_name: str,
    restart_pipeline: bool,
    overwrite_model: bool,
    subset_size: float,
):
    """
    data (Pandas Dataframe): requires at least the following columns
        'html' -> html as a string
        'label_name' -> label as a Boolean
    labels (list): the label names as strings
    """
    print("---Starting test process!")
    print("---Text Extraction!")
    data_path_text = fw.get_dataset_checkpoint_path("text")
    if restart_pipeline or not os.path.exists(data_path_text):
        print("\t-Reading datafile!")
        df = pd.read_csv(fw.get_dataset_path())
        df = df.sample(int(subset_size * len(df)))
        df["html"] = df["path_to_html"].apply(load_html)

        print("\t-Extracting Text!")
        text_extractor = TextExtraction()
        df["html"] = df["html"].apply(text_extractor.extract_text_from_html)
        df = df.drop(df[df["html"].map(len) == 0].index)
        df = df.drop_duplicates(subset=["html"] + labels)
        df.to_csv(data_path_text, index=False, escapechar="\\")
        print("\t-Number of samples", len(df))
    else:
        print("\t-Text already has been extracted!")
        df = pd.read_csv(data_path_text)

    print("---Preprocess Text!")
    data_path_prepped = fw.get_dataset_checkpoint_path("prepped")
    if restart_pipeline or not os.path.exists(data_path_prepped):
        print("\t-Preprocessing data!")
        text_preprocessor = TextPreprocessor()
        df["html"] = df["html"].apply(text_preprocessor.preprocess_data)
        df = df.drop(df[df["html"].map(len) == 0].index)
        df = df.drop_duplicates(subset=["html"] + labels)
        print("\t-Number of remainig samples after deduplication", len(df))
        df.to_csv(data_path_prepped, index=False, escapechar="\\")
    else:
        print("\t-Text has already been preprocessed!")
        df = pd.read_csv(data_path_prepped)

    print("---Vectorizing Text!")
    data_path_vectorized = fw.get_dataset_checkpoint_path("vectorized", add_label=True)
    if restart_pipeline or not os.path.exists(data_path_vectorized):
        print("\t-Applying vectorization to data!")
        text_vectorizer = TextVectorization()
        X = text_vectorizer.vectorize(df["html"].values)
        y = df.loc[:, labels].values
        new_df = pd.DataFrame(np.concatenate((X, y), axis=1))
        new_df.to_csv(data_path_vectorized, index=False)
    else:
        print("\t-Text has already been vectorized!")
        df_vec = pd.read_csv(data_path_vectorized)
        y = df_vec.iloc[:, -len(labels) :].values
        X = df_vec.iloc[:, : -len(labels)].values

    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(
        X, y, train_frac=0.7, val_frac=0.2, test_frac=0.1
    )
    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)
    print(X_test.shape, y_test.shape)

    # Train model
    data = [X_train, X_val, X_test, y_train, y_val, y_test, labels]
    model_save_path = fw.get_model_path(model_name)
    clf_bert = ClassificationManagerSBERT(data, model_save_path, overwrite_model, mute=True)
    clf_bert.train_model(50)
    results = clf_bert.evaluate_model()
    print(results)
    print('Average precision: ', results['samples avg']['precision'])


def main():
    experiment_name = "test"
    dataset_name = "ganymede_data_mini"
    model_name = "test"
    overwrite_dataset = False  # Allow dataset to be overwritten
    overwrite_results = False  # Allow experiment results to be overwritten
    overwrite_model = True  # Allow model results to be overwritten
    restart_pipeline = False  # Force execution of all steps in the pipeline
    subset_size = 0.05  # Fraction of total samples to be used

    fw = FileWarehouse(experiment_name, dataset_name, overwrite_results)

    abuse_labels = [
        "Cybercrime",
        "Drugs / Narcotics",
        "Financial Crime",
        "Goods and Services",
        "Sexual Abuse",
        "Violent Crime",
    ]

    # Create dataset
    create_ganymede_dataset(
        fw=fw, labels=abuse_labels, overwrite_dataset=overwrite_dataset, deduplication_method="title", n_folders=10
    )
    train_sbert(fw, abuse_labels, model_name, restart_pipeline, overwrite_model, subset_size)

    


if __name__ == "__main__":
    main()
