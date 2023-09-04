import os
import json
import random

from src.data.setup_experiment import FileWarehouse

random.seed(11)
from typing import List

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder


def load_html(html_file_path):
    with open(html_file_path, errors="replace") as f:
        return f.read()


def select_scrape_folders(main_data_folder: str, n_scrapes: int):
    scrape_folders = os.listdir(main_data_folder)

    accepted_folders = []
    for folder in scrape_folders:
        if folder.startswith("Scrape") and (250000 <= int(folder.split("_")[1]) < 1100000):
            accepted_folders.append(folder)

    n_scrapes = n_scrapes if 1 <= n_scrapes <= len(accepted_folders) else len(accepted_folders)
    selected_folders = random.sample(accepted_folders, n_scrapes)
    return selected_folders


def create_ganymede_dataset(
    fw: FileWarehouse,
    labels: List[str],
    overwrite_dataset: bool,
    deduplication_method: str = None,
    n_folders: int = 1,
):
    dataset_path = fw.get_dataset_path()
   
    if not overwrite_dataset and os.path.exists(dataset_path):
        print("A dataset with this name already exists!")
        return

    # Setup pandas
    base_columns = ["domain_id", "page_id", "page_version_id", "path_to_html", "content_hash", "title"]
    lb = LabelEncoder()
    lb.fit(labels)
    df = pd.DataFrame(columns=base_columns + list(lb.classes_))

    # Start scrape
    base_data_path = "/home/pieter/TimeLine/Django/Data/"
    scrape_folders = select_scrape_folders(base_data_path, n_folders)
    for scrape in scrape_folders:
        print(scrape)

        # Load page version file
        page_version_info_path = os.path.join(base_data_path, scrape, "pages_versions.json")
        with open(page_version_info_path) as f:
            page_version_info = json.load(f)
        pages_first_versions = [version for version in page_version_info if version["page_version"] == 0]

        # Load domain info file
        domain_info_path = os.path.join(base_data_path, scrape, "domain.json")
        with open(domain_info_path) as f:
            domain_info = json.load(f)

        assert len(domain_info) == len(pages_first_versions)

        # Populate dataset
        for page, domain in zip(pages_first_versions, domain_info):
            domain_id = domain["group"].split("-")[0]
            page_id = page["id"]
            page_version_id = page["html_file_path"].split("/")[1].split("-")[0]

            if "title" in domain and "English" in domain["tag_list"]:
                html_file_path = os.path.join(base_data_path, scrape, page["html_file_path"] + ".html")
                if not os.path.exists(html_file_path):
                    continue

                label_union = set(domain["tag_list"]) & set(labels)
                if label_union:
                    target_idx = lb.transform(list(label_union))
                    target_vector = np.zeros(len(labels))
                    target_vector[target_idx] = 1

                elif "No abuse" in domain["tag_list"]:
                    target_vector = np.zeros(len(labels))

                df.loc[len(df)] = [
                    domain_id,
                    page_id,
                    page_version_id,
                    html_file_path,
                    page["content_hash"],
                    domain["title"],
                ] + list(target_vector)

    if deduplication_method:
        df = df.drop_duplicates(subset=[deduplication_method])

    df = df.drop(["content_hash", "title"], axis=1)
    df.to_csv(dataset_path, index=False)

    dataset_config = {
        "scraped_folders": scrape_folders,
        "n_folders": n_folders,
        "base_columns": base_columns,
        "data_path": dataset_path,
        "labels": labels,
    }

    dataset_config_path = fw.get_dataset_config_path()
    with open(dataset_config_path, "w") as f:
        json.dump(dataset_config, f, indent=4)
