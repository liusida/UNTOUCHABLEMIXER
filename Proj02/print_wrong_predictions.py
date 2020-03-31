# This file if for manually checking the wrong predictions.
import copy
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
JarvisBigBrain = VotingClassifier( estimators=brains, voting="hard" )
JarvisBigBrain2 = copy.copy(JarvisBigBrain)

# Preprocess training data
data_x, data_y, in_filenames = utils.preprocess_data(with_filename=True)

train_x, test_x, train_y, test_y = train_test_split(data_x, data_y)

# Cross evaluate all possible candidates
import glob
problematic_file_counts = {}
allfiles = glob.glob(utils.training_folders[1]+"/*")
print("Evaluation start.")
print(f"Size of total data: {len(data_y)}.")
JarvisBigBrain.fit(train_x, train_y)
data_y_hat = JarvisBigBrain.predict(data_x)
wrong_predictions = (data_y!=data_y_hat)
for i in range(len(wrong_predictions)):
    if wrong_predictions[i]:
        print(f"{data_x[i]}|| true:[{data_y[i]}] => prediction:[{data_y_hat[i]}]")
        file = in_filenames[i]
        if file in problematic_file_counts:
            problematic_file_counts[file] += 1
        else:
            problematic_file_counts[file] = 1

print("in files:")
ids = utils.sorted_dict(problematic_file_counts, desc=True)
for f in ids:
    print(f)

# in this list, we found those buggy files:
# and we are going to delete them and use the rest of the data to train a brain to win the competition 1!
#
# training_data/external-data/176ea1b1-6426-4cd5-1ea4-5cd69371a71f
# training_data/external-data/8c1745a7-9a6a-5f92-cca7-4147f6be1f72
# training_data/external-data/d3447490-96fd-35d0-adf2-0806e5214606
# training_data/external-data/0870e15c-2fcd-81b5-d24b-ace4307bf326
# training_data/external-data/1fdb8b32-06d5-99e8-12f1-75ffae3b16ec
# training_data/external-data/1846d424-c17c-6279-23c6-612f48268673
# training_data/external-data/11ebcd49-428a-1c22-d5fd-b76a19fbeb1d
# training_data/external-data/8d723104-f773-83c1-3458-a748e9bb17bc
# training_data/external-data/e3e70682-c209-4cac-629f-6fbed82c07cd
# training_data/external-data/12e0c8b2-bad6-40fb-1948-8dec4f65d4d9
# training_data/external-data/a25b59fd-92e8-e269-d12e-cbc40b9475b1