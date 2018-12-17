## Algorithms
 - [XGBoost Classifier](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBClassifier)

## Class distributions

| Data       | Positive | Negative |
|------------|----------|----------|
| Up-selling | 3682     | 46318    |
| Churn      | 3672     | 46328    |
| Appetency  | 890      | 49110    |

## Best results

### Nominal columns transformed into categories

| Data       | Max Depth | #Estimators | Subsample | Score  | AUC                |
|------------|-----------|-------------|-----------|--------|--------------------|
| Up-selling | 3         | 100         | 1.0       | 0.9458 | 0.7169399228911428 |
| Churn      | 3         | 100         | 1.0       | 0.926  | 0.502758595094867  |
| Appetency  | 3         | 100         | 0.9       | 0.98   | 0.5                |

*Using 10-fold CV*

## Instruction

See Instructions.md

## Usage

```sh
$ python main.py -k 10 --max_depth 3 --n_estimators 100 --subsample 1.0 -l data\labels\orange_small_train_upselling.labels
$ python main.py -k 10 --max_depth 3 --n_estimators 100 --subsample 1.0 -l data\labels\orange_small_train_churn.labels
$ python main.py -k 10 --max_depth 3 --n_estimators 100 --subsample 0.9 -l data\labels\orange_small_train_appetency.labels
```