from datetime import datetime
from pathlib import Path
from random import gauss
from unittest import result
from sklearnex import patch_sklearn, config_context
patch_sklearn()
import imblearn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import (
    decomposition,
    discriminant_analysis,
    ensemble,
    linear_model,
    metrics,
    model_selection,
    naive_bayes,
    pipeline,
    preprocessing,
    svm,
)

from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA


np.set_printoptions(precision=3, suppress=True)
pd.options.display.float_format = "{:,.3g}".format

DATA = Path("public_data")

PRED_PATH = Path("Submissions")

DROP_VARS = ["ADMITTIME", "DISCHTIME", "SUBJECT_ID", "HADM_ID"]

assert DATA.is_dir()

features = pd.read_csv(
    DATA / "mimic_synthetic_feat.name", header=None
).values.flatten()

labels = pd.read_csv(
    DATA / "mimic_synthetic_label.name", header=None
).values.flatten()

x_df = pd.read_csv(
    DATA / "mimic_synthetic_train.data",
    header=None,
    names=features,
    sep=" ",
)

# Remove variables that are not relevant
x_df.drop(columns=DROP_VARS, inplace=True)

ys = pd.Series(
    pd.read_csv(
        DATA / "mimic_synthetic_train.solution",
        header=None,
        names=labels,
        sep=" ",
    ).values.flatten()
)

# Load test set
x_test_df = pd.read_csv(
    DATA / "mimic_synthetic_test.data",
    header=None,
    names=features,
    sep=" ",
)

# Remove variables that are not relevant
x_test_df.drop(columns=DROP_VARS, inplace=True)

###############################################################################

x_nans = x_df.isna().sum()
x_miss = x_nans[x_nans > 0]
x_test_nans = x_test_df.isna().sum()
x_test_miss = x_test_nans[x_test_nans > 0]

na_cols = set(x_miss.index) | set(x_test_miss.index)

for col in na_cols:
    x_df[col].fillna(x_df[col].mode()[0], inplace=True)
    x_test_df[col].fillna(x_test_df[col].mode()[0], inplace=True)

###############################################################################

x_all_oh_df = pd.get_dummies(pd.concat([x_df, x_test_df]))

x_all_oh_df.loc[:,'ys'] = ys

x_all_oh_df.apply(lambda x: pd.factorize(x)[0])

# x_all_oh_df.drop((col for col in x_all_oh_df if len(x_all_oh_df[col].unique()) == 1), axis="columns", inplace=True)


def cross_val ():
    SCORINGS = "balanced_accuracy"

    pipe = make_pipeline(PCA(n_components=12), GaussianNB())

    scores = model_selection.cross_val_score(pipe, x_oh_df.loc[:, x_oh_df.columns != 'ys'], x_oh_df['ys'], cv=10, scoring=SCORINGS)

    with np.printoptions(precision=2):
        print(scores)

    print(f"\n{SCORINGS}: {scores.mean():.2f}, with std dev: {scores.std():.2f}\n")


print(x_all_oh_df.shape)

from sklearn.feature_selection import VarianceThreshold
constant_filter = VarianceThreshold(threshold=0)
constant_filter.fit(x_all_oh_df)
constant_columns = [column for column in x_all_oh_df.columns
                    if column not in x_all_oh_df.columns[constant_filter.get_support()]]

x_all_oh_df.drop(labels=constant_columns, axis=1, inplace=True)
print(x_all_oh_df.shape)

qconstant_filter = VarianceThreshold(threshold=0.01)
qconstant_filter.fit(x_all_oh_df)
qconstant_columns = [column for column in x_all_oh_df.columns
                    if column not in x_all_oh_df.columns[qconstant_filter.get_support()]]

x_all_oh_df.drop(labels=qconstant_columns, axis=1, inplace=True)

x_oh_df = x_all_oh_df.iloc[: len(x_df)].copy()
x_test_oh_df = x_all_oh_df.iloc[len(x_df):].copy()

correlated_features = set()
correlation_matrix = x_oh_df.corr()

for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.7:
            colname = correlation_matrix.columns[i]
            correlated_features.add(colname)


x_oh_df.drop(labels=correlated_features, axis=1, inplace=True)
x_test_oh_df.drop(labels=correlated_features, axis=1, inplace=True)

from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import LogisticRegression

x_train, x_test, y_train, y_test = train_test_split(
    x_oh_df.drop(['ys'], axis=1),
    x_oh_df['ys'],
    test_size=0.2,
    random_state=7
)

from sklearn.neighbors import NearestCentroid

from sklearn.metrics import balanced_accuracy_score

# for nbc in range(1, 131):
pipe = make_pipeline(RobustScaler(), PCA(), NearestCentroid())

pipe.fit(x_train, y_train)
Y_pca_pred = pipe.predict(x_test)
resultat = balanced_accuracy_score(y_test, Y_pca_pred)
print(resultat)



# kfold = KFold(n_splits=5)
# model = LogisticRegression(C=7.7, solver='saga')
# results = model_selection.cross_val_score(model, x_oh_df.drop(['ys'], axis=1), x_oh_df['ys'])
# print(results)
# print("Accuracy: ", results.mean()*100)

# parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
# svc = svm.SVC()
# clf = GridSearchCV(svc, parameters)

# param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid']}
# grid = GridSearchCV(svm.SVC(),param_grid,refit=True,verbose=2)
# grid.fit(x_train,y_train)
# print(grid.best_estimator_)

# logistic = svm.SVC()

# penalty = ['l1', 'l2']

# C = np.logspace(0, 4, 10)

# hyperparameters = dict(C=C, penalty=penalty)

# clf = GridSearchCV(logistic, hyperparameters, cv=5, verbose=0)
# best_model=clf.fit(robust_df.drop(['ys'], axis=1), robust_df['ys'])

# print("Best penalty:", best_model.best_estimator_.get_params()['penalty'])
# print("Best C:", best_model.best_estimator_.get_params()['C'])























# print(x_oh_df.shape)

# cross_val()

# def cross_val_v2 ():
#     SCORINGS = "balanced_accuracy"

#     pipe = imblearn.pipeline.Pipeline(
#     [
#         ("scale", preprocessing.RobustScaler()),
#         ("pca", decomposition.PCA(n_components=12)),
#         ("resample", imblearn.over_sampling.SMOTE()),
#         ("model", GaussianNB()),
#     ]
#     )

#     scores = model_selection.cross_val_score(pipe, x_oh_df.loc[:, x_oh_df.columns != 'ys'], x_oh_df['ys'], cv=10, scoring=SCORINGS)

#     with np.printoptions(precision=2):
#         print(scores)

#     print(f"\n{SCORINGS}: {scores.mean():.2f}, with std dev: {scores.std():.2f}\n")

# cross_val_v2()



# from numpy import arange
# from sklearn.feature_selection import VarianceThreshold

# thresholds = arange(0.0, 0.55, 0.05)
# for t in thresholds:
#     x_oh_df = VarianceThreshold(threshold=t).fit_transform(x_oh_df)
#     print(x_oh_df.shape)
#     cross_val()

# accuracy_scores=np.zeros((130,2))

# for index, nbc in enumerate(range(1, 131)):
#     pipe = imblearn.pipeline.Pipeline(
#     [
#         ("scale", preprocessing.RobustScaler()),
#         ("pca", decomposition.PCA(n_components=nbc)),
#         ("resample", imblearn.over_sampling.SMOTE()),
#         ("model", GaussianNB()),
#     ]
#     )
#     scores = model_selection.cross_val_score(pipe, x_oh_df.loc[:, x_oh_df.columns != 'ys'], x_oh_df['ys'], cv=10, scoring="balanced_accuracy")
#     accuracy_scores[index, 1]=( f"{scores.mean():.4f}")
#     accuracy_scores[index, 0]=nbc

# print(pd.DataFrame(accuracy_scores,columns=["nb_component","accuracy_score"]).sort_values(ascending = True, by=["accuracy_score"]))

# accuracy_scores=np.zeros((131,2))

# for index, nbc in enumerate(range(1, 131)):
#     pipe = imblearn.pipeline.Pipeline(
#     [
#         ("scale", preprocessing.RobustScaler()),
#         ("pca", decomposition.PCA(n_components=nbc)),
#         ("model", GaussianNB()),
#     ]
#     )
#     scores = model_selection.cross_val_score(pipe, x_oh_df.loc[:, x_oh_df.columns != 'ys'], x_oh_df['ys'], cv=10, scoring="balanced_accuracy")
#     accuracy_scores[index, 1]=( f"{scores.mean():.4f}")
#     accuracy_scores[index, 0]=nbc

# accuracy_scores=np.zeros((131,2))

# for index, nbc in enumerate(range(1, 130)):
#     pipe = imblearn.pipeline.Pipeline(
#     [
#         ("scale", preprocessing.StandardScaler()),
#         ("pca", decomposition.PCA(n_components=nbc)),
#         ("model", GaussianNB()),
#     ]
#     )
#     scores = model_selection.cross_val_score(pipe, x_oh_df.loc[:, x_oh_df.columns != 'ys'], x_oh_df['ys'], cv=10, scoring="balanced_accuracy")
#     accuracy_scores[index, 1]=( f"{scores.mean():.4f}")
#     accuracy_scores[index, 0]=nbc

# print(pd.DataFrame(accuracy_scores,columns=["nb_component","accuracy_score"]).sort_values(ascending = True, by=["accuracy_score"]))

# accuracy_scores=np.zeros((131,2))

# for index, nbc in enumerate(range(1, 130)):
#     pipe = imblearn.pipeline.Pipeline(
#     [
#         ("scale", preprocessing.StandardScaler()),
#         ("pca", decomposition.PCA(n_components=nbc)),
#         ("model", linear_model.LogisticRegression(max_iter=10000)),
#     ]
#     )
#     scores = model_selection.cross_val_score(pipe, x_oh_df.loc[:, x_oh_df.columns != 'ys'], x_oh_df['ys'], cv=10, scoring="balanced_accuracy")
#     accuracy_scores[index, 1]=( f"{scores.mean():.4f}")
#     accuracy_scores[index, 0]=nbc

# print(pd.DataFrame(accuracy_scores,columns=["nb_component","accuracy_score"]).sort_values(ascending = True, by=["accuracy_score"]))

# accuracy_scores=np.zeros((131,2))

# for index, nbc in enumerate(range(1, 130)):
#     pipe = imblearn.pipeline.Pipeline(
#     [
#         ("scale", preprocessing.StandardScaler()),
#         ("pca", decomposition.PCA(n_components=nbc)),
#         ("model", linear_model.LogisticRegression(max_iter=10000)),
#     ]
#     )
#     scores = model_selection.cross_val_score(pipe, x_oh_df.loc[:, x_oh_df.columns != 'ys'], x_oh_df['ys'], cv=10, scoring="balanced_accuracy")
#     accuracy_scores[index, 1]=( f"{scores.mean():.4f}")
#     accuracy_scores[index, 0]=nbc

# print(pd.DataFrame(accuracy_scores,columns=["nb_component","accuracy_score"]).sort_values(ascending = True, by=["accuracy_score"]))

# 191           192           0.618
# 190           191           0.627
# 217           218            0.63
# 216           217           0.638
# 218           219           0.656