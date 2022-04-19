from datetime import datetime
from pathlib import Path
from random import gauss
from turtle import exitonclick
from unittest import result
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


np.set_printoptions(precision=3, suppress=True)
pd.options.display.float_format = "{:,.3g}".format

DATA = Path("public_data")

PRED_PATH = Path("Submissions")

DROP_VARS = ["ADMITTIME", "DISCHTIME", "SUBJECT_ID", "HADM_ID","Allergy"]

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



ys = pd.Series(
    pd.read_csv(
        DATA / "mimic_synthetic_train.solution",
        header=None,
        names=labels,
        sep=" ",
    ).values.flatten()
)



n=0.001
### merge var
s = x_df.corrwith(ys[:], axis=0)

s = s[s<n]
s = s[s>0]
mergeVar=[]
for x in s.index:
    mergeVar.append(x)
print(len(mergeVar))
x_df['lessCorr']=x_df[mergeVar[0]]
for var in mergeVar:
        x_df['lessCorr']=x_df['lessCorr']+x_df[var]

# Remove variables that are not relevant
x_df.drop(columns=DROP_VARS, inplace=True)
try:
    x_df.drop(columns=mergeVar, inplace=True)
except Exception as e :
    pass


### merge varNeg
s = x_df.corrwith(ys[:], axis=0)

s = s[s>-n]
s = s[s<0]
mergeVarNeg=[]
for x in s.index:
    mergeVarNeg.append(x)
len(mergeVarNeg)
x_df['lessCorrNeg']=x_df[mergeVarNeg[0]]
for var in mergeVarNeg:
        x_df['lessCorrNeg']=x_df['lessCorrNeg']+x_df[var]


try:
    x_df.drop(columns=mergeVarNeg, inplace=True)
except Exception as e :
    pass










# Load test set
x_test_df = pd.read_csv(
    DATA / "mimic_synthetic_test.data",
    header=None,
    names=features,
    sep=" ",
)




# Remove variables that are not relevant
x_test_df.drop(columns=DROP_VARS, inplace=True)
try:
    x_test_df.drop(columns=mergeVar, inplace=True)
except Exception as e :
    pass

try:
    x_test_df.drop(columns=mergeVarNeg, inplace=True)
except Exception as e :
    pass
x_test_df['lessCorr']=x_df['lessCorr']
x_test_df['lessCorrNeg']=x_df['lessCorrNeg']





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

X = x_oh_df.drop(['ys'], axis=1)
Y = x_oh_df['ys']


from sklearn.neighbors import KNeighborsClassifier, NearestCentroid

from sklearn.metrics import balanced_accuracy_score


def cross_val ():
    SCORINGS = "balanced_accuracy"

    pipe = make_pipeline(PCA(n_components=0.90), NearestCentroid())

    scores = model_selection.cross_val_score(pipe, X, Y, cv=10, scoring=SCORINGS)

    with np.printoptions(precision=2):
        print(scores)

    print(f"\n{SCORINGS}: {scores.mean():.2f}, with std dev: {scores.std():.2f}\n")



print(x_oh_df.shape)

x_train, x_test, y_train, y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=7
)


accuracy_scores=np.zeros((118,2))

# for index, nbc in enumerate(range(1, 119)):
#     pipe = imblearn.pipeline.Pipeline(
#     [
#         ("scale", StandardScaler()),
#         ("pca", PCA(n_components=nbc)),
#         ("resample", imblearn.over_sampling.SMOTE()),
#         ("model", NearestCentroid()),
#     ]
#     )
#     pipe.fit(x_train, y_train)
#     pred_test_rbt = pipe.predict(x_test)
#     resultat = balanced_accuracy_score(y_test, pred_test_rbt)
#     accuracy_scores[index, 1]=( f"{resultat :.04f}")
#     accuracy_scores[index, 0]=nbc

print(pd.DataFrame(accuracy_scores,columns=["nb_component","accuracy_score"]).sort_values(ascending = True, by=["accuracy_score"]))

x_train, x_test, y_train, y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=7
)

## drop rows 

# ## y_test
# y_train_died = y_train[y_train==1].index
# nbdied= len(y_train_died)*5
# y_train_lived = y_train[y_train==0].index
# y_train_live_save = y_train_lived[0:nbdied]
# y_train=y_train_live_save.append(y_train_died)
# ## x_train
# x_train.drop(y_train_lived[nbdied-1:-1],inplace=True)
# print(x_train.shape)
# print(y_train.shape)



names = [
    # "Nearest Neighbors",
    # "Linear SVM",
    "RBF SVM",
    # "Gaussian Process",
    # "Decision Tree",
    # "Random Forest",
    # "Neural Net",
    # "AdaBoost",
    # "Naive Bayes",
    # "QDA",
]

classifiers = [
    # KNeighborsClassifier(3),
    # svm.SVC(kernel="linear", C=1.0),
    svm.SVC(gamma=2, C=1),
    # GaussianProcessClassifier(),
    # DecisionTreeClassifier(max_depth=5),
    # ensemble.RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    # MLPClassifier(alpha=1, max_iter=1000),
    # ensemble.AdaBoostClassifier(),
    # GaussianNB(),
    # discriminant_analysis.QuadraticDiscriminantAnalysis(),
]


    


# for name,model in zip(names,classifiers):
#     print(name)
#     pipe = imblearn.pipeline.Pipeline(
#         [
#             ("scale", StandardScaler()),
#             ("pca", PCA(n_components=0.90)),
#             ("resample", imblearn.over_sampling.SMOTE()),
#             ("model", model),
#         ]
#     )

#     print(x_train.shape)
#     try:
#         pipe.fit(x_train, y_train)
#         Y_pca_pred = pipe.predict(x_test)
#     except Exception as e :
#         pass
#     resultat = balanced_accuracy_score(y_test, Y_pca_pred)
#     print(resultat)

for i in range(10):
    print(i)
    pipe = imblearn.pipeline.Pipeline(
        [
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=0.90)),
            ("resample", imblearn.over_sampling.SMOTE()),
            ("model", svm.SVC(gamma=i+1, C=i+1)),
        ]
    )

    print(x_train.shape)

    pipe.fit(x_train, y_train)
    Y_pca_pred = pipe.predict(x_test)
    resultat = balanced_accuracy_score(y_test, Y_pca_pred)
    print(resultat)

# pipe = imblearn.pipeline.Pipeline(
#     [
#         ("scale", StandardScaler()),
#         ("pca", PCA(n_components=118)),
#         ("resample", imblearn.over_sampling.SMOTE()),
#         ("model", NearestCentroid()),
#     ]
# )

# pipe.fit(x_train, y_train)
# Y_pca_pred = pipe.predict(x_test)
# resultat = balanced_accuracy_score(y_test, Y_pca_pred)
# print(resultat)

# pipe.fit(X, Y)

# predictions = pipe.predict(x_test_oh_df.drop(['ys'], axis=1))

# PRED_PATH.mkdir(parents=True, exist_ok=True)

# t_stamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
# submission_fp = PRED_PATH / f"submission_{t_stamp}.zip"

# pred_fname = "mimic_synthetic_test.csv"
# compr_opts = dict(method="zip", archive_name=pred_fname)

# pd.Series(predictions).to_csv(
#     submission_fp, compression=compr_opts, index=False, header=False
# )

# print(f"The submission is ready: {submission_fp}")

# CV = 10
# SCORINGS = "balanced_accuracy"

# pipe = make_pipeline(RobustScaler(), PCA(), NearestCentroid())

# scores = model_selection.cross_val_score(pipe, X, Y, cv=10, scoring=SCORINGS)

# with np.printoptions(precision=2):
#     print(scores)

# print(f"\n{SCORINGS}: {scores.mean():.2f}, with std dev: {scores.std():.2f}\n")

# pipe = make_pipeline(StandardScaler(), PCA(), NearestCentroid())

# scores = model_selection.cross_val_score(pipe, X, Y, cv=10, scoring=SCORINGS)

# with np.printoptions(precision=2):
#     print(scores)

# print(f"\n{SCORINGS}: {scores.mean():.2f}, with std dev: {scores.std():.2f}\n")

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