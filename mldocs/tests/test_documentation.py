import json
import pytest
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from shutil import rmtree
from mldocs.documentation import Documentation


SAVE_PATH = 'mldocs/tests/fixtures/tmp/'


class Helper(object):

    def __init__(self):
        pass

    @staticmethod
    def environment_initialize():
        df_x, df_y = load_iris(return_X_y=True)
        x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, random_state=0)

        lg = LogisticRegression()
        lg.fit(x_train, y_train)

        documenter = Documentation(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, model=lg, random_state=0,
                            metrics={'accuracy': (metrics.accuracy_score, {}),
                                     'precision': (metrics.precision_score, {'average': 'micro'})},
                            save_dir=SAVE_PATH, comment='this is an example usage', problem_kind='classification')

        documenter.document()

        return documenter.docu


    @staticmethod
    def find_diff(dict1, dict2):
        differences = []
        for key in dict1.keys():
            if type(dict1[key]) is dict:
                return Helper.find_diff(dict1[key], dict2[key])
            else:
                if not dict1[key] == dict2[key]:
                    differences.append((key, dict1[key], dict2[key]))
        return differences



class TestDocumentation(object):

    def setup_class(cls):
        cls.doc = Helper.environment_initialize()
        with open('mldocs/tests/fixtures/fixture_docu_iris-logreg.json', 'r') as file:
            cls.truth = json.load(file)


    def test_keys(self):
        assert len(set(self.doc.keys()).symmetric_difference(set(self.truth))) == 0


    def test_dataset(self):
        print(self.truth['dataset'])
        print(self.doc['dataset'])
        diff = Helper.find_diff(self.truth['dataset'], self.doc['dataset'])
        print(diff)
        assert len(diff) == 0


    def test_model(self):
        diff = Helper.find_diff(self.truth['model'], self.doc['model'])
        assert len(diff) == 0


    def test_performance(self):
        diff = Helper.find_diff(self.truth['performance'], self.doc['performance'])
        left, right = [x[1] for x in diff], [x[1] for x in diff]
        assert pytest.approx(left, 0.01) == right

    def teardown_class(cls):
        rmtree(SAVE_PATH)
