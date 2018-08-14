from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from shutil import rmtree
from mldocs.documentation import Documentation


SAVE_PATH = ''


def run():
    df_x, df_y = load_iris(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, random_state=0)

    lg = LogisticRegression()
    lg.fit(x_train, y_train)

    doc = Documentation(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, model=lg, random_state=0,
                        metrics={'accuracy': (metrics.accuracy_score, {}), 'precision': (metrics.precision_score, {'average':'micro'})},
                        save_dir=SAVE_PATH, comment='this is an example usage', problem_kind='classification')

    doc.document()


if __name__ == "__main__":
    run()
    #rmtree(SAVE_PATH)
