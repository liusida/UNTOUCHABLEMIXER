# Running this file will generate a Jarvis brain pickle file
# At the end of competition 1, we will submit the generated pickle file.

# =======================================================================
# Important Constant
#
# This filename is specified in Sec 5.1 Proj 02
brain_filename = "jarvis_UNTOUCHABLEMIXER.pkl"
# This orde is specified in Sec 7.2 Proj 01
targets = ["TIME", "PIZZA", "GREET", "WEATHER", "JOKE"]
#========================================================================

import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

JarvisBrain = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])

data_x = targets
data_y = list(range(len(targets)))

JarvisBrain.fit(data_x, data_y)

with open(brain_filename, "wb") as f:
    pickle.dump(JarvisBrain, f)
