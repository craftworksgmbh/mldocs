import numpy as np
import pandas as pd
import os
import json
from shutil import rmtree
from multiprocessing import Pool
from time import gmtime, strftime, time
from sklearn.externals import joblib
from scipy import stats


KERAS = 'keras'
SKLEARN = 'sklearn'
CSV_EXT = '.csv'
X_TRAIN = 'x_train'
Y_TRAIN = 'y_train'
X_TEST = 'x_test'
Y_TEST = 'y_test'
TRAIN = 'train'
TEST = 'test'
TIMESTAMP = 'timestamp'
DATASET = 'dataset'
MODEL = 'model'
PERFORMANCE = 'performance'
N_ROWS = 'n_rows'
N_FEATURES = 'n_features'
FEATURES = 'features'
N_TARGET = 'n_target'
TARGET = 'target'
PARAMETERS = 'parameters'
OPTIMIZER = 'optimizer'
LOSS = 'loss'
NAME = 'name'
OPTIMIZER_PARAMETERS = 'optimizer_parameters'
H5_EXT = '.h5'
PKL_EXT = '.pkl'
METRICS = 'metrics'
PREDICTION_TIME_PER_SAMPLE = 'pred_time_per_sample_in_sec'
JSON_EXT = '.json'
DOCU = 'docu'
COMMENT = 'comment'
KIND = 'kind'
ONEHOTENCODED = 'one_hot_encoding'
PROBLEM_KIND = 'problem_kind'
CLASSIFICATION = 'classification'
REGRESSION = 'regression'
TARGET_INFO = 'target_info'
RANDOM_STATE = 'random_state'
MEAN = 'mean'
MEDIAN = 'median'
MODE = 'mode'
STD = 'std'
MIN = 'min'
MAX = 'max'
N_CLASSES = 'n_classes'
CLASS_FREQUENCIES = 'class_frequencies'
CONFIG = 'config'

# TODO: notebook and checkpoint saving
# TODO: refactor


class Documentation(object):
    '''
    Class that ensures reproducibility of machine learning pipelines by saving state:
     - train data
     - test data
     - model
     - model parameters
     - time
     - performance
    '''

    def __init__(self, x_train: np.array, y_train: np.array, x_test: np.array, y_test: np.array, model: object,
                 random_state: int, metrics: dict, save_dir: str, problem_kind: str, comment: str = '',
                 nthreads: int = 1):
        '''
        Documentation constructor.

        :param x_train: train features
        :param y_train: train labels
        :param x_test: test features
        :param y_test: test labels
        :param model: fitted model (keras, sklearn)
        :param random_state: integer of random state used for training
        :param metrics: dict of tuples of metrics that should be assessed: {'name_metric1': (metric_function, kwargs)}
        :param save_dir: directory where timestamp folder should be created and documentation should be saved
        :param problem_kind: classification/regression
        :param comment: free text field
        :param nthreads: number of threads used for saving data (multiprocessing)
        '''

        self.x_train = pd.DataFrame(x_train)
        self.y_train = pd.DataFrame(y_train)
        self.x_test = pd.DataFrame(x_test)
        self.y_test = pd.DataFrame(y_test)
        self.data_descr = [(X_TRAIN, self.x_train), (Y_TRAIN, self.y_train),
                           (X_TEST, self.x_test), (Y_TEST, self.y_test)]
        self.model = model
        self.random_state = random_state
        self.metrics = metrics
        self.timestamp = strftime("%d-%m-%Y_%H-%M-%S", gmtime())
        self.save_dir = save_dir + self.timestamp + '/'
        self.comment = comment
        self.nthreads = nthreads
        self.kind = self._get_model_kind()
        self.problem_kind = problem_kind
        self.docu = self._populate_base_docu()
        self._validate_input()

    def _validate_input(self):
        '''
        Validates input.

        :return: None
        '''

        if not any([True for kind in [SKLEARN, KERAS] if kind in str(type(self.model))]):
            raise TypeError('Model must be of type keras or sklearn.')

        if not isinstance(self.random_state, int):
            raise TypeError('Random state must be integer.')

        if not isinstance(self.metrics, dict):
            raise TypeError('Metrics must be of type dict')
        else:
            if not isinstance(self.metrics[list(self.metrics.keys())[0]], tuple):
                raise TypeError('Each value of dict metrics must be a tuple where the first entry is a sklearn.metrics'
                                'function and the second is a dictionary of kwargs to that function.')

        if not isinstance(self.save_dir, str):
            raise TypeError('save_dir must be of type string')

        if not any([True for item in [REGRESSION, CLASSIFICATION] if item == self.problem_kind]):
            raise ValueError('problem_kind must be one of {"classification", "regression"}')

        if not isinstance(self.nthreads, int):
            raise TypeError('nthreads must be integer')


    def _create_directory(self):
        '''
        Creates directory structure needed.

        :return: None
        '''

        if os.path.exists(self.save_dir):
            self._cleanup()
        os.makedirs(self.save_dir, exist_ok=True)


    def _populate_base_docu(self):
        '''
        Creates basic structure of documentation dictionary.

        :return: dictionary structure as dict
        '''

        docu = dict()
        docu[RANDOM_STATE] = self.random_state
        docu[TIMESTAMP] = self.timestamp
        docu[DATASET] = {
            TRAIN: {
                X_TRAIN: dict(),
                Y_TRAIN: dict(),
            },
            TEST: {
                X_TEST: dict(),
                Y_TEST: dict(),
            }
        }
        docu[MODEL] = dict()
        docu[PERFORMANCE] = {
            METRICS: dict()
        }
        docu[COMMENT] = self.comment
        docu[PROBLEM_KIND] = self.problem_kind

        return docu

    def _get_model_kind(self):
        '''
        Checks if model is keras or sklearn model.

        :return: string indicating whether it is keras or sklearn model
        '''

        if KERAS in str(type(self.model)):
            return KERAS
        elif SKLEARN in str(type(self.model)):
            return SKLEARN

    def _save_data(self):
        '''
        Saves data used for training and testing using multiprocessing. (master)

        :return: None
        '''

        with Pool(self.nthreads) as pool:
            pool.map(self._save_data_worker, self.data_descr)

    def _save_data_worker(self, data: list):
        '''
        Saves data used for training and testing using multiprocessing. (worker)

        :param data: list of tuples like (name: str, df: pd.DataFrame); e.g. ('x_train', x_train)
        :return: None
        '''

        name, df = data
        df.to_csv(self.save_dir + name + CSV_EXT)

    def _describe_data(self):
        '''
        Extracts information on data and adds it to documentation dictionary.

        :return: None
        '''

        for name, df in self.data_descr:
            n_rows = len(df)
            n_cols = len(df.columns)
            columns = list(df.columns)

            if TRAIN in name:
                dataset_type = TRAIN
                exact_type = Y_TRAIN
                this_df = self.y_train
            elif TEST in name:
                dataset_type = TEST
                exact_type = Y_TEST
                this_df = self.y_test

            self.docu[DATASET][dataset_type][name][N_ROWS] = n_rows
            self.docu[DATASET][dataset_type][name][N_FEATURES] = n_cols
            self.docu[DATASET][dataset_type][name][FEATURES] = columns

            if self.problem_kind == CLASSIFICATION:
                if this_df.shape[-1] > 1:
                    self.docu[DATASET][dataset_type][exact_type][ONEHOTENCODED] = True
                    self.docu[DATASET][dataset_type][exact_type][N_CLASSES] = this_df.shape[-1]
                    unique, counts = np.unique(np.argmax(this_df.values, 0), return_counts=True)

                else:
                    self.docu[DATASET][dataset_type][exact_type][ONEHOTENCODED] = False
                    self.docu[DATASET][dataset_type][exact_type][N_CLASSES] = len(list(np.unique(this_df)))
                    unique, counts = np.unique(this_df.values, return_counts=True)

                unique = [str(item) for item in unique]
                counts = [int(item) for item in counts]
                self.docu[DATASET][dataset_type][exact_type][CLASS_FREQUENCIES] = dict(zip(unique, counts))

            elif self.problem_kind == REGRESSION:
                self.docu[DATASET][dataset_type][exact_type][MEAN] = np.mean(this_df)
                self.docu[DATASET][dataset_type][exact_type][MEDIAN] = np.median(this_df)
                self.docu[DATASET][dataset_type][exact_type][MODE] = stats.mode(this_df)
                self.docu[DATASET][dataset_type][exact_type][MIN] = np.min(this_df)
                self.docu[DATASET][dataset_type][exact_type][MAX] = np.max(this_df)
                self.docu[DATASET][dataset_type][exact_type][STD] = np.std(this_df)

    def _get_model_setup(self):
        '''
        Extracts setup of model and adds it to documentation dictionary.

        :return: None
        '''

        if self.kind == KERAS:
            self.docu[MODEL][KIND] = self.model.name
            self.docu[MODEL][PARAMETERS] = dict()
            self.docu[MODEL][PARAMETERS][CONFIG] = self.model.get_config()
            self.docu[MODEL][PARAMETERS][OPTIMIZER] = dict()
            self.docu[MODEL][PARAMETERS][OPTIMIZER][NAME] = str(self.model.optimizer).split('.')[2]
            self.docu[MODEL][PARAMETERS][OPTIMIZER][OPTIMIZER_PARAMETERS] = self.model.optimizer.get_config()
            self.docu[MODEL][PARAMETERS][LOSS] = self.model.loss
        elif self.kind == SKLEARN:
            self.docu[MODEL][KIND] = str(self.model).split('(')[0]
            self.docu[MODEL][PARAMETERS] = self.model.get_params()

    def _save_model(self):
        '''
        Saves model.

        :return: None
        '''

        model_save_path = self.save_dir + self.docu[MODEL][KIND]
        if self.kind == KERAS:
            self.model.save(model_save_path + H5_EXT)
        elif self.kind == SKLEARN:
            joblib.dump(self.model, model_save_path + PKL_EXT)

    def _get_performance(self):
        '''
        Measures performance and speed of fitted model on test set (prediction).

        :return: None
        '''

        start_time = time()
        preds = self.model.predict(self.x_test.values)
        elapsed = (time() - start_time)
        self.docu[PERFORMANCE][PREDICTION_TIME_PER_SAMPLE] = elapsed / self.docu[DATASET][TEST][X_TEST][N_ROWS]

        for name, metric_info in self.metrics.items():
            metric_fun, params = metric_info
            self.docu[PERFORMANCE][METRICS][name] = metric_fun(self.y_test.values, preds, **params)

    def _save_documentation(self):
        '''
        Saves documentation directory to disk.

        :return: None
        '''

        with open(self.save_dir + DOCU + JSON_EXT, 'w') as file:
            json.dump(self.docu, file)

    def _cleanup(self):
        '''
        Recursively force deletes directory.

        :return: None
        '''

        rmtree(self.save_dir, ignore_errors=True)

    def document(self):
        '''
        Sequentially executes documentation steps.

        :return: None
        '''

        try:
            self._create_directory()
            self._save_data()
            self._describe_data()
            self._get_model_setup()
            self._save_model()
            self._get_performance()
            self._save_documentation()
        except Exception as err:
            self._cleanup()
            raise err
