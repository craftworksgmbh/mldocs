# Documenting the Machine Learning Process



## Example Usage

```
from mldocs.documentation import Documentation
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


df_x, df_y = load_iris(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, random_state=0)

lg = LogisticRegression()
lg.fit(x_train, y_train)

doc = Documentation(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, model=lg, random_state=0,
                    metrics={'accuracy': (metrics.accuracy_score, {}), 'precision': (metrics.precision_score, {'average':'micro'})},
                    save_dir=SAVE_PATH, comment='this is an example usage', problem_kind='classification')

doc.document()

```

## Output (JSON)

```
{
  "timestamp": "14-08-2018_14-04-34",
  "dataset": {
    "test": {
      "y_test": {
        "one_hot_encoding": false,
        "features": [
          0
        ],
        "class_frequencies": {
          "0": 13,
          "2": 9,
          "1": 16
        },
        "n_classes": 3,
        "n_rows": 38,
        "n_features": 1
      },
      "x_test": {
        "n_rows": 38,
        "n_features": 4,
        "features": [
          0,
          1,
          2,
          3
        ]
      }
    },
    "train": {
      "y_train": {
        "one_hot_encoding": false,
        "features": [
          0
        ],
        "class_frequencies": {
          "0": 37,
          "2": 41,
          "1": 34
        },
        "n_classes": 3,
        "n_rows": 112,
        "n_features": 1
      },
      "x_train": {
        "n_rows": 112,
        "n_features": 4,
        "features": [
          0,
          1,
          2,
          3
        ]
      }
    }
  },
  "model": {
    "kind": "LogisticRegression",
    "parameters": {
      "dual": false,
      "fit_intercept": true,
      "verbose": 0,
      "class_weight": null,
      "max_iter": 100,
      "random_state": null,
      "solver": "liblinear",
      "n_jobs": 1,
      "intercept_scaling": 1,
      "multi_class": "ovr",
      "tol": 0.0001,
      "C": 1.0,
      "warm_start": false,
      "penalty": "l2"
    }
  },
  "performance": {
    "metrics": {
      "precision": 0.868421052631579,
      "accuracy": 0.868421052631579
    },
    "pred_time_per_sample_in_sec": 3.9276323820415294e-06
  },
  "comment": "this is an example usage",
  "problem_kind": "classification",
  "random_state": 0
}

```
