filename = "input.txt"


# This file if for manually checking the wrong predictions.
import copy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ParameterGrid
from sklearn.ensemble import VotingClassifier

# Models we will use:
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import NuSVC
# Models end

import utils

# All the models we are going to use
candidate_models = [
    MultinomialNB,
    BernoulliNB,
    DecisionTreeClassifier,
    ExtraTreeClassifier,
    KNeighborsClassifier,
    LogisticRegression,
    NearestCentroid,
    RandomForestClassifier,
    RidgeClassifier,
    NuSVC,
]
# The parameters that could pass to model
candidate_parameters = {
    "LogisticRegression":       {"solver": ["lbfgs"], "multi_class": ["auto"]},
    "RandomForestClassifier":   {"n_estimators": [10,30]},
    "NuSVC":                    {"gamma": ["scale"]},
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
JarvisBigBrain = VotingClassifier( estimators=brains, voting="hard" )

# Preprocess training data
data_x, data_y, in_filenames = utils.preprocess_data(with_filename=True)
JarvisBigBrain.fit(data_x, data_y)

test_x, test_y = utils.read_one_file("input.txt")
score = JarvisBigBrain.score(test_x, test_y)

if score>0.85:
    print("good")
else:
    print("bad")
