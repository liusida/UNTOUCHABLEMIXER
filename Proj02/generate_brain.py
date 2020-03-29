# Running this file will generate a Jarvis brain pickle file
# At the end of competition 1, we will submit the generated pickle file.

# =======================================================================
# Important Constant
#
# This filename is specified in Sec 5.1 Proj 02
brain_filename = "jarvis_UNTOUCHABLEMIXER.pkl"
# This folder name contains all the data files that will feed into the JarvisBrain
training_folder = "training_data"
# These labels are specified in Sec 7.2 Proj 01
targets = ["TIME", "PIZZA", "GREET", "WEATHER", "JOKE"]
#========================================================================

import pickle, glob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

import utils

# Build the model
JarvisBrain = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])

# Preprocess training data
data_x = targets.copy()
data_y = targets.copy()
training_filenames = glob.glob(training_folder+"/*")
for filename in training_filenames:
    with open(filename, "r") as f:
        lines = f.readlines()
    for line in lines:
        if line[0]=="#":
            # it is a comment line
            continue
        if len(line.strip()) == 0:
            # it is an empty line
            continue
        try:
            strings = line.split(",")
            if len(strings)==2:
                data_x.append(strings[0].strip())
                data_y.append(strings[1].strip())
            elif len(strings)>2:
                sentence = ",".join(strings[:-1])
                data_x.append(sentence.strip())
                data_y.append(strings[-1].strip())
            else:
                utils.error(f"Format error in line: {line}.")
        except:
            utils.error(f"Processing line: {line}.")

# Visually check the data
if False:
    print(data_x[:10])
    print("===")
    print(data_y[:10])

# Train
JarvisBrain.fit(data_x, data_y)

# Save as a pickle file
with open(brain_filename, "wb") as f:
    pickle.dump(JarvisBrain, f)
