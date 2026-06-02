"""This script is used to prepare pickle objects that log all the info relative to a specific experimental run. This
was supposed to be an ad-hoc implementation that would work readily for GRIMP, so it is not refined at all. It was
"good enough" for the experiments.

The paths used by the logger can be provided by the user, or can use the default values used here.

Raises:
    ValueError: _description_
    ValueError: _description_

Returns:
    _type_: _description_
"""
from pathlib import Path
import pickle
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import os.path as osp
import os
import json

import pandas as pd
from abc import abstractmethod, ABC

# Setting the default path for all files. Assumes that the dir tree is already built.
RESULTS_PATH = osp.realpath("grimpresults")
PLOTS_PATH = osp.realpath("grimpresults/plots")
JSON_PATH = osp.join(RESULTS_PATH, "json")
# The run_id file should contain only a number, which is incremented automatically
# at the start of each run.
RUN_ID_PATH = osp.realpath("grimpdata/run_id")


class Logger(ABC):
    def __init__(
        self,
        file_path=None,
        run_id_path=RUN_ID_PATH,
        results_path=RESULTS_PATH,
        plots_path=PLOTS_PATH,
    ):

        # All paths below are normally kept as default, but can be provided by
        # the user.
        self.run_id_path = run_id_path
        self.run_name = None
        self.results_path = results_path
        self.plots_path = plots_path

        self.run_id = self.find_latest_run_id()
        if file_path is None:
            # If no pre-existing path is provided, create a new empty logger file.
            self.obj = dict()
            self.obj["run_id"] = self.run_id
            self.obj["timestamps"] = dict()
            self.obj["durations"] = dict()

            # Losses and actual imputation results
            self.obj["results"] = dict()
            # Statistics measured on the given dataset (% missing values etc)
            self.obj["statistics"] = dict()

            self.add_time("logger_creation_time")

            # Ensure that the required folders exist
            os.makedirs("grimpresults", exist_ok=True)
            os.makedirs("grimpresults/plots", exist_ok=True)
        else:
            self.obj = pickle.load(open(file_path, "rb"))
            self.run_id = self.obj["run_id"]

    def find_latest_run_id(self):
        """Utility function for opening the run_id file, checking for errors and
        incrementing it by one at the start of a run.

        Raises:
            ValueError: Raise ValueError if the read run_id is not a positive integer.

        Returns:
            int: The new (incremented) run_id.
        """
        if osp.exists(self.run_id_path):
            with open(self.run_id_path, "r") as fp:
                last_run_id = fp.read().strip()
                try:
                    run_id = int(last_run_id) + 1
                except ValueError:
                    raise ValueError(
                        f"Run ID {last_run_id} is not a positive integer. "
                    )
                if run_id < 0:
                    raise ValueError(f"Run ID {run_id} is not a positive integer. ")
            with open(self.run_id_path, "w") as fp:
                fp.write(f"{run_id}")
        else:
            run_id = 0
            with open(self.run_id_path, "w") as fp:
                fp.write(f"{run_id}")
        return run_id

    def add_dict(self, obj_name, obj):
        """
        Add a new dictionary to the logger. `obj_name` is the key that will be used to store the object.

        :param obj_name: A string that will be used as key.
        :param obj: The dictionary to be added.
        :return:
        """
        self.obj[obj_name] = dict()
        self.obj[obj_name].update(obj)

    def update_dict(self, obj_name, obj):
        """Update the given object with a new dictionary.

        Args:
            obj_name (str): Label of the object to update.
            obj (dict): Dictionary to be added to the given obj.
        """
        self.obj[obj_name].update(obj)

    def add_value(self, obj_name, key, value):
        """Updating a single value in a given object.

        Args:
            obj_name (str): Label of the object to update.
            key (_type_): Key of the object to update.
            value (_type_): Value of the object to update.
        """
        self.obj[obj_name][key] = value

    def add_run_name(self, name):
        """Generic function for setting a run name provided by the user.

        Args:
            name (str): Name to be assigned to this run.
        """
        self.obj["parameters"]["run_name"] = name
        self.run_name = name

    def get_value(self, obj_name, key):
        """Retrieve a single value from one of the dictionaries.

        Args:
            obj_name (str): Label of the object to query.
            key (_type_): Dict key to use to retrieve the value.

        Returns:
            _type_: Retrieved value.
        """
        return self.obj[obj_name][key]

    def add_time(self, label):
        """Add a new timestamp starting __now__, with the given label.

        Args:
            label (str): Label to assign to the timestamp.
        """
        self.obj["timestamps"][label] = dt.datetime.now()

    def get_time(self, label):
        """Retrieve a time according to the given label.

        Args:
            label (str): Label of the timestamp to be retrieved.
        Returns:
            _type_: Retrieved timestamp.
        """
        return self.obj["timestamp"][label]

    def add_duration(self, label_start, label_end, label_duration):
        """Create a new duration as timedelta. The timedelta is computed on the
        basis of the given start and end labels, and is assigned the given label.

        Args:
            label_start (str): Label of the timestamp to be used as start.
            label_end (str): Label of the timestamp to be used as end.
            label_duration (str): Label of the new timedelta object.
        """
        assert label_start in self.obj["timestamps"]
        assert label_end in self.obj["timestamps"]

        self.obj["durations"][label_duration] = (
            self.obj["timestamps"][label_end] - self.obj["timestamps"][label_start]
        ).total_seconds()

    def save_obj(self, file_path=None):
        """Save log object in a specific file_path, if provided. Alternatively,
        save the log object in a default location.

        Args:
            file_path (str, optional): Path where to save the log file. Defaults to None.
        """
        if file_path:
            if not osp.exists(file_path):
                raise ValueError(f"File {file_path} does not exist.")
            pickle.dump(self.obj, open(file_path, "wb"))
        else:
            file_path = osp.join(self.results_path, f"run_{self.run_id}.pkl")
            pickle.dump(self.obj, open(file_path, "wb"))

    def __getitem__(self, item):
        return self.obj[item]

    @abstractmethod
    def pprint(self):
        """Abstract class, this should be implemented by the user with the proper
        format for the task at hand.
        """
        pass

    @abstractmethod
    def get_header():
        """Abstract class, this should be implemented by the user with the proper
        header for the task at hand. This header is relative to the rolling
        results logging file.
        """
        pass

    def print_selected(self, selected_dict):
        # TODO: this function is not done
        for obj_dict in selected_dict:
            selected_entries = selected_dict[obj_dict]
            for entry in selected_entries:
                val = self.obj[entry][entry]

    def update_result_file(self):
        """Update the result file with the string produced by pprint. If the file
        does not exist, create it, then update it.
        """
        if osp.exists(osp.join(self.results_path, "grimpresults.csv")):
            with open(osp.join(self.results_path, "grimpresults.csv"), "a") as fp:
                fp.write(self.pprint())
        else:
            header = self.get_header()
            df = pd.DataFrame(columns=header)
            df.to_csv(osp.join(self.results_path, "grimpresults.csv"), index=False)
            with open(osp.join(self.results_path, "grimpresults.csv"), "a") as fp:
                fp.write(self.pprint())


class GrimpLogger(Logger):
    def add_run_name(self):
        """Set the default run name, based on the name of the dirty dataset (note that the name includes the error
        fraction in the given dataset.)
        """
        basename = osp.basename(self.obj["parameters"]["dirty_dataset"])
        name, ext = osp.splitext(basename)
        self.obj["parameters"]["run_name"] = name
        self.run_name = name

    def pprint(self):
        """Implementation of the pprint method. The statistics and parameters
        are all over the place because of backwards compatibility with previous
        code.

        Returns:
            str: Pretty-printed output of the logger.
        """
        self.summary = ""
        values = [
            self.run_id,
            "GRIMP-ML",
            self.obj["parameters"]["dirty_dataset"],
            self.obj["parameters"]["training_columns"],
            self.obj["statistics"]["training_rows"],
            self.obj["parameters"]["epochs"],
            self.obj["parameters"]["predictor_structure"],
            self.obj["parameters"]["gnn_structure"],
            self.obj["statistics"]["unidirectional"],
            self.obj["parameters"]["architecture"],
            self.obj["parameters"]["no_relu"],
            self.obj["parameters"]["no_sm"],
            self.obj["parameters"]["k_strat"],
            self.obj["parameters"]["learning_rate"],
            self.obj["parameters"]["weight_decay"],
            self.obj["parameters"]["module_aggr"],
            self.obj["parameters"]["dropout_gnn"],
            self.obj["parameters"]["dropout_clf"],
            self.obj["parameters"]["flag_col"],
            self.obj["statistics"]["comb_num"],
            self.obj["statistics"]["comb_size"],
            "|".join(self.obj["statistics"]["node_features"]),
            self.obj["parameters"]["max_components"],
            self.obj["statistics"]["training_columns"],
            # self.obj['statistics']['dtype_columns'],
            # 'num_fds',
            self.obj["statistics"]["num_fds"],
            self.obj["statistics"]["fd_strategy"],
            self.obj["statistics"]["fd_path"],
            self.obj["statistics"]["num_rows"],
            self.obj["statistics"]["num_distinct_values"],
            self.obj["statistics"]["num_missing_values"],
            self.obj["durations"]["training_duration"],
            self.obj["results"]["imp_accuracy"],
            self.obj["results"]["tot_true"],
            self.obj["curves"]["min"],
            self.obj["curves"]["end"],
            self.obj["curves"]["min_valid"],
            self.obj["curves"]["end_valid"],
        ]
        if "accuracy_dict" in self.obj["results"]:
            for col in self.obj["results"]["accuracy_dict"]:
                values.append(self.obj["results"]["accuracy_dict"][col])

        s = ",".join([str(_) for _ in values])
        return s + "\n"

    def plot_curves(self):
        """Plot the training/validation loss curves for the given run. Minimum
        and final values for both losses is also added to the plots. Plots are
        saved in the folder indicated by self.plots_path.
        """
        loss = self["curves"]["loss"]
        loss_valid = self["curves"]["loss_valid"]
        fig = plt.Figure()
        ax = fig.gca()
        (p1,) = plt.plot(loss)
        (p11,) = plt.plot(loss_valid)
        (p2,) = plt.plot(np.argmin(loss), np.min(loss), marker="o")
        (p22,) = plt.plot(np.argmin(loss_valid), np.min(loss_valid), marker="+")
        (p3,) = plt.plot(len(loss), loss[-1], marker="x")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        short_name = osp.basename(self.obj["parameters"]["dirty_dataset"])
        plt.title(f"Run {self.obj['run_id']} - Dataset {short_name}")
        plt.legend(
            [p1, p11, p2, p22, p3],
            ["loss", "loss_valid", "min", "min_valid", "end"],
            loc="upper left",
        )
        plt.savefig(osp.join(self.plots_path, f"{self.run_id}.png"))

    def print_summary(self):
        """Print on screen a summary of this run, with the main parameters."""
        print(f"Run ID:{self.run_id}")
        print(f"Dataset: {self.obj['parameters']['dirty_dataset']}"),
        print(f"Training columns: {self.obj['parameters']['training_columns']}")
        print(f"Total epochs: {self.obj['parameters']['epochs']}")
        print(f"Architecture: {self.obj['parameters']['architecture']}")
        print(f"Loss function: {self.obj['parameters']['loss']}")
        print(f"Node features: {self.obj['statistics']['node_features']}")

    @staticmethod
    def get_header():
        """Return the default value of the header for the results.csv file.

        Returns:
            str: Prepared header.
        """
        header = [
            "run_id",
            "algorithm",
            "dirty_dataset",
            "training_columns",
            "training_rows",
            "epochs",
            "predictor_structure",
            "gnn_structure",
            "unidirectional",
            "architecture",
            "no_relu",
            "no_sm",
            "k_strat",
            "learning_rate",
            "weight_decay",
            "aggregation_function",
            "dropout_gnn",
            "dropout_clf",
            "cid_flag",
            "comb_num",
            "comb_size",
            "node_features",
            "max_components",
            "training_columns",
            "num_fds",
            "fd_strategy",
            "fd_path",
            "num_rows",
            "num_distinct_values",
            "num_missing_values",
            "training_duration",
            "imputation_accuracy",
            "total_correct_imputations",
            "min_loss",
            "final_loss",
            "min_valid_loss",
            "final_valid_loss",
        ]

        header = header + [f"imp_col_{i}" for i in range(1, 21)]

        return header

    def save_obj(self, file_path=None, plot_figures=False):
        """Save the log package in the given file, update the results file with
        the results from the current run and plot figures if needed.

        Args:
            file_path (str, optional): Path to the log package. Defaults to None.
            plot_figures (bool, optional): Flag that decides whether to plot losses or not. Defaults to False.
        """
        super(GrimpLogger, self).save_obj(file_path)
        super(GrimpLogger, self).update_result_file()
        if plot_figures:
            self.plot_curves()

    def save_json(self):
        result_dict = {
            "dataset_name": self.run_name,
            "imputation_method": "GRIMP",
            "run_params": {
                k: str(v) if isinstance(v, Path) else v
                for k, v in self.obj["parameters"].items()
            },
            "start_time": self.obj["timestamps"]["start_training"].isoformat(),
            "end_time": self.obj["timestamps"]["end_training"].isoformat(),
            "exec_time": self.obj["durations"]["duration_training"],
        }

        fpath = f"{self.run_name}_grimp.json"
        ofp = open(osp.join(JSON_PATH, fpath), "w")
        json.dump(result_dict, ofp, indent=2)
        ofp.close()

    def add_df_stats(self, df):
        """Simple function for adding the dataframe statistics to the logger.

        Args:
            df (pd.DataFrame): _description_
        """
        self.obj["statistics"]["num_rows"] = len(df)
        num_distinct_values = set(df.values.ravel().tolist())
        self.obj["statistics"]["num_distinct_values"] = num_distinct_values
        num_missing_values = df.isna().sum().sum()
        self.obj["statistics"]["num_missing_values"] = num_missing_values


def logging(parameters, results):
    logger = Logger()
    logger.add_dict("parameters", parameters)
    logger.add_dict("results", results)
