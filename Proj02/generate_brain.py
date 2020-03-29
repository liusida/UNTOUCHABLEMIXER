# Running this file will generate a Jarvis brain pickle file
# At the end of competition 1, we will submit the generated pickle file.

brain_filename = "jarvis_UNTOUCHABLEMIXER.pkl"

import pickle

JarvisBrain = ["empty"]

with open(brain_filename, "wb") as f:
    pickle.dump(JarvisBrain, f)
