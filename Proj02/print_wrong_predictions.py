# This file evaluates several models and hyper-parameters in sklearn

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ParameterGrid
from sklearn.ensemble import VotingClassifier # Voting will help us win the competition! We can combine the top models together!

# Models we will use: (TODO: Read the documentation and come up with better Classifiers and Parameters)
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestCentroid
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import NuSVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import AdaBoostClassifier
# Models end

import utils

# All the models we are going to use
candidate_models = [
    AdaBoostClassifier,
    SGDClassifier,
    MultinomialNB,
    BernoulliNB,
    DecisionTreeClassifier,
    ExtraTreeClassifier,
    ExtraTreesClassifier,
    KNeighborsClassifier,
    LinearSVC,
    LogisticRegression,
    NearestCentroid,
    RandomForestClassifier,
    RidgeClassifier,
    NuSVC,
    GradientBoostingClassifier,
]
# The parameters that could pass to model
candidate_parameters = {
    "SGDClassifier":            {"max_iter": [10, 100], "tol" : [1e-3], "loss": ["log", "modified_huber"]}, 
    "ExtraTreesClassifier":     {"n_estimators": [20, 200]},
    "LinearSVC":                {"max_iter": [200, 2000], "multi_class": ["crammer_singer"]},
    "LogisticRegression":       {"solver": ["lbfgs"], "multi_class": ["auto"]},
    "RandomForestClassifier":   {"n_estimators": [10,30]},
    "NuSVC":                    {"gamma": ["scale", "auto"]},
}
# Build the model
brains = []
for this_model in candidate_models:
    if this_model.__name__ in candidate_parameters:
        parameter_grid = candidate_parameters[this_model.__name__]
        parameters = list(ParameterGrid(parameter_grid)) # make grid for parameter-pairs
        for i, parameter in enumerate(parameters):
            brain = Pipeline([
                ("vect", CountVectorizer()),
                ("tfidf", TfidfTransformer()),
                ("clf", this_model(**parameter)),
            ])
            brains.append( (f"{this_model.__name__}-{i}", brain) )
    else:
        brain = Pipeline([
            ("vect", CountVectorizer()),
            ("tfidf", TfidfTransformer()),
            ("clf", this_model()),
        ])
        brains.append( (f"{this_model.__name__}", brain) )

# Prepare models for testing
testing_brains = brains.copy()

# Preprocess training data
data_x, data_y = utils.preprocess_data()

# Visually check the data
if False:
    print(data_x[:10])
    print("===")
    print(data_y[:10])

train_x, test_x, train_y, test_y = train_test_split(data_x, data_y)

# Cross evaluate all possible candidates
print("Evaluation start.")
print(f"Size of total data: {len(data_y)}.")
names = []
cv_score = []
cv_scores = []
timers = []

for brain in testing_brains:
    utils.start_timer()
    print(f"Training Model name: {brain[0]}", flush=True)
    brain[1].fit(train_x, train_y)
    print("")

    test_y_hat = brain[1].predict(test_x)
    wrong_predictions = (test_y_hat!=test_y)
    j = 0
    for i in range(len(wrong_predictions)):
        if wrong_predictions[i]:
            print(f"{test_x[i]} true:[{test_y[i]}] => prediction:[{test_y_hat[i]}]")
            j+=1
            if j>20:
                break
