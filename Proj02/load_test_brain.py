# This file load the generated Jarvis brain pickle file and evaluate it.

# =======================================================================
# Important Constant
#
# This filename is specified in Sec 5.1 Proj 02
brain_filename = "jarvis_UNTOUCHABLEMIXER.pkl"
# This orde is specified in Sec 7.2 Proj 01
targets = ["TIME", "PIZZA", "GREET", "WEATHER", "JOKE"]
#========================================================================

import pickle
brain = pickle.load(open(brain_filename, 'rb'))
result = brain.predict(["Hello funny roboooot!"])
for ret in result:
    print(targets[ret])
