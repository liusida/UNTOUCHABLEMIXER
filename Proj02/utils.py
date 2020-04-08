# This file contains all the utilities that make life easier
#

# =======================================================================
# Important Constant
#
# This filename is specified in Sec 5.1 Proj 02
brain_filename = "jarvis_UNTOUCHABLEMIXER.pkl"
# This folder name contains all the data files that will feed into the JarvisBrain
training_folders = ["training_data/external-data"]
cleaned_training_folders = [ "training_data/cleaned-data" ]
# These labels are specified in Sec 7.2 Proj 01
targets = ["TIME", "PIZZA", "GREET", "WEATHER", "JOKE"]

#Jarvis database created in Proj 01
db_file = "jarvis.db"
#========================================================================

#========================================================================
# Useful Tools
# output error
def error(error_message=""):
    return print('\033[31m ERROR: ' + '\033[0m' + error_message)
# timer
import time
timer = 0
def start_timer():
    global timer
    timer = time.time()
def stop_timer():
    return time.time() - timer
# sort a dictionary by value
def sorted_dict(dictionary, desc=False):
    if desc:
        return sorted(dictionary.items(), key=lambda x: x[1])[::-1]
    else:
        return sorted(dictionary.items(), key=lambda x: x[1])

#========================================================================

#========================================================================
# Reuseable functionalities
def preprocess_data(cleaned_version=True, with_filename=False):
    '''
    cleaned_version: True if we use manually cleaned version for competition, False if we use training_folders.
    with_filename: True if we keep the filename in the list, so that we can figure out which file the data comes from. False by default.
    '''
    import glob, json, os
    data_x = targets.copy()
    data_y = targets.copy()
    in_filenames = [""]*len(targets) # just padding some empty strings, to make it aligned with data_x and data_y.
    if cleaned_version:
        target_folders = cleaned_training_folders
    else:
        target_folders = training_folders
    for training_folder in target_folders:
        training_filenames = glob.glob(training_folder+"/*")
        for filename in training_filenames:
            if os.path.isdir(filename):
                # skip sub-folders
                continue
            with open(filename, "r", encoding="UTF-8") as f:
                lines = f.readlines()
            for line in lines:
                if line[0]=="#":
                    # it is a comment line
                    continue
                if len(line.strip()) == 0:
                    # it is an empty line
                    continue
                if line[0]=="{":
                    # it is a json line
                    try:
                        j = json.loads(line)
                        data_x.append(j["TXT"].strip())
                        data_y.append(j["ACTION"].strip())
                        if with_filename:
                            in_filenames.append(filename)
                    except:
                        error(f"Json loads error. line: {line}.")
                else:
                    try:
                        strings = line.split(",")
                        if len(strings)==2:
                            data_x.append(strings[0].strip())
                            data_y.append(strings[1].strip())
                            if with_filename:
                                in_filenames.append(filename)
                        elif len(strings)>2:
                            sentence = ",".join(strings[:-1])
                            data_x.append(sentence.strip())
                            data_y.append(strings[-1].strip())
                            if with_filename:
                                in_filenames.append(filename)
                        else:
                            error(f"Format error. line: {line}.")
                    except:
                        error(f"Split by comma error. line: {line}.")
    # Check validation of the target
    target_set = set(targets)
    for i in data_y:
        if not i in target_set:
            error(f"Invalid target value: {i}.")
    if with_filename:
        return data_x, data_y, in_filenames
    else:
        return data_x, data_y


#Function to read in database from Project 1
def project1_data():
    ''' 
    Function to read in database created in Project 1 when first creating Jarvis.
    The databast should be specified using db_file. 
    Outputs two lists:
    data_x: all string commands included in database
    data_y: corresponding action targets for Jarvis
    '''
    import sqlite3
    #data phrases
    data_x = []
    #Data targets/actions
    data_y = []
    db_conn = sqlite3.connect(db_file)
    c = db_conn.cursor()
    
    for line in c.execute('SELECT * FROM training_data'):
        data_x.append(line[0].strip())
        data_y.append(line[1].strip())
    return data_x, data_y


def build_a_simple_brain():
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import RidgeClassifier
    brain = Pipeline([
            ("vect", CountVectorizer()),
            ("tfidf", TfidfTransformer()),
            ("clf", RidgeClassifier()),
        ])
    return brain

def build_a_complex_brain():
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
    # All the models we are going to use
    candidate_models = [
        AdaBoostClassifier,
        # SGDClassifier,
        MultinomialNB,
        BernoulliNB,
        DecisionTreeClassifier,
        ExtraTreeClassifier,
        ExtraTreesClassifier,
        KNeighborsClassifier,
        # LinearSVC,
        LogisticRegression,
        NearestCentroid,
        RandomForestClassifier,
        RidgeClassifier,
        # NuSVC,
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
    brain = VotingClassifier( estimators=brains, voting="hard" )
    return brain
