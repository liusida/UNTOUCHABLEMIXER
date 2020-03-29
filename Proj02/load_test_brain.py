# This file load the generated Jarvis brain pickle file and evaluate it.

brain_filename = "jarvis_UNTOUCHABLEMIXER.pkl"

import pickle

with open(brain_filename, "rb") as f:
    JarvisBrain = pickle.load(f)

print(JarvisBrain)