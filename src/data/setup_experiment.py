import os
import json


class FileWarehouse:
    def __init__(
        self,
        experiment_name: str,
        dataset_name: str,
        overwrite_results: bool,
        base_folder: str = os.getcwd(),

    ):
        self.experiment_name = experiment_name
        self.dataset_name = dataset_name
        self.label_name = ""

        # Create folder names
        self.data_folder = os.path.join(base_folder, "data")
        self.results_folder = os.path.join(base_folder, "results")
        self.datasets_folder = os.path.join(self.data_folder, "datasets")
        self.checkpoints_folder = os.path.join(self.datasets_folder, f"checkpoints_{experiment_name}")
        self.model_folder = os.path.join(self.data_folder, "models")
        self.experiment_result_folder = os.path.join(self.results_folder, experiment_name)

        # Create folders if they dont exist
        for path in [
            self.data_folder,
            self.results_folder,
            self.datasets_folder,
            self.checkpoints_folder,
            self.model_folder,
            self.experiment_result_folder,
        ]:
            if not os.path.exists(path):
                os.mkdir(path)

        if not overwrite_results and os.listdir(self.experiment_result_folder):
            raise FileExistsError("An experiment with this name already exists")

    def get_dataset_path(self):
        return os.path.join(self.datasets_folder, f"{self.dataset_name}.csv")

    def get_dataset_config_path(self):
        return os.path.join(self.datasets_folder, f"{self.dataset_name}_config.json")

    def get_dataset_checkpoint_path(self, checkpoint_name: str, add_label: bool = False):
        if add_label:
            return os.path.join(self.checkpoints_folder, f"{self.dataset_name}_{checkpoint_name}_{self.label_name}.csv")
        else:
            return os.path.join(self.checkpoints_folder, f"{self.dataset_name}_{checkpoint_name}.csv")

    def get_model_path(self, model_name: str):
        return os.path.join(self.model_folder, f"{model_name}.pt")

    def get_model_config_path(self, model_name: str):
        return os.path.join(self.model_folder, f"{model_name}_config.json")

    def get_result_predictions_path(self, model_name: str):
        return os.path.join(self.experiment_result_folder, f"{model_name}_classification_preds.csv")

    def get_result_report_path(self, model_name: str):
        return os.path.join(self.experiment_result_folder, f"{model_name}_classification_report.json")

    def create_experiment_config_file(self, config):
        config_path = os.path.join(self.experiment_result_folder, "experiment_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)
