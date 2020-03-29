# This file load the generated Jarvis brain pickle file and evaluate it.

# =======================================================================
# Important Constant
#
# This filename is specified in Sec 5.1 Proj 02
brain_filename = "jarvis_UNTOUCHABLEMIXER.pkl"
#========================================================================

import pickle
brain = pickle.load(open(brain_filename, 'rb'))
result = brain.predict(["Hello funny roboooot!", "TEMPERATURE"])
print(result)