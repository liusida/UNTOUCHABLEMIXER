# This file contains all the utilities that make life easier
#

# =======================================================================
# Important Constant
#
# This filename is specified in Sec 5.1 Proj 02
brain_filename = "jarvis_UNTOUCHABLEMIXER.pkl"
# This folder name contains all the data files that will feed into the JarvisBrain
training_folders = ["training_data", "training_data/external-data"]
# These labels are specified in Sec 7.2 Proj 01
targets = ["TIME", "PIZZA", "GREET", "WEATHER", "JOKE"]
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
#========================================================================

#========================================================================
# Reuseable functionalities
def preprocess_data():
    import glob, json, os
    data_x = targets.copy()
    data_y = targets.copy()
    for training_folder in training_folders:
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
                    except:
                        error(f"Json loads error. line: {line}.")
                else:
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
                            error(f"Format error. line: {line}.")
                    except:
                        error(f"Split by comma error. line: {line}.")
    # Check validation of the target
    target_set = set(targets)
    for i in data_y:
        if not i in target_set:
            error(f"Invalid target value: {i}.")
    return data_x, data_y
