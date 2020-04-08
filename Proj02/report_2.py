import glob, shutil, os
import numpy as np
import matplotlib.pyplot as plt
import utils
from sklearn.model_selection import train_test_split

external_data_files = "training_data/external-data/*"
cleaned_data_folder = "training_data/cleaned-data"

# pick GOOD training data from external-data folder and copy them to cleaned-data folder
def step1():
    print("Step 1 start. Pick GOOD files from external-data to cleaned-data.")
    # make a folder for cleaned data
    if os.path.exists(cleaned_data_folder):
        shutil.rmtree(cleaned_data_folder)
    os.mkdir(cleaned_data_folder)

    # get a brain
    brain = utils.build_a_complex_brain()
    # get all external-data
    data_x, data_y, in_filenames = utils.preprocess_data(cleaned_version=False, with_filename=True)
    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y)
    print(f"Size of total data: {len(data_y)}.")
    # count data in each file
    all_file_counts = {}
    problematic_file_counts = {}
    for key in in_filenames:
        if key=="":
            continue
        if key in all_file_counts:
            all_file_counts[key] += 1
        else:
            all_file_counts[key] = 1
        if key not in problematic_file_counts:
            problematic_file_counts[key] = 0
    # count problematic data in each file
    allfiles = glob.glob(external_data_files)
    brain.fit(train_x, train_y)
    data_y_hat = brain.predict(data_x)
    wrong_predictions = (data_y!=data_y_hat)
    for i in range(len(wrong_predictions)):
        if wrong_predictions[i]:
            # print(f"{data_x[i]}|| true:[{data_y[i]}] => prediction:[{data_y_hat[i]}]")
            file = in_filenames[i]
            problematic_file_counts[file] += 1

    # report the result and copy data to cleaned folder
    print("Petential errors in files:")
    ids = utils.sorted_dict(problematic_file_counts, desc=True)
    prefix = "training_data/external-data/"
    for filename, wrong_num in ids:
        short_filename = filename[len(prefix):]
        percentage = 100 * wrong_num / all_file_counts[filename]
        should_exclude = False
        if percentage>30:
            should_exclude = True
        print(f"{short_filename}: ({wrong_num}/{all_file_counts[filename]}, {percentage:.1f}%) {'recommend to exclude' if should_exclude else ''}")
        if not should_exclude:
            shutil.copyfile(filename, f"{cleaned_data_folder}/{short_filename}")

# hold off test data and gradually adding more training data to the brain
def step2():
    print("Step 2 start. Show difference of adding external data.")
    # get externel-data from cleaned folder
    proj1_data_x, proj1_data_y = utils.project1_data()
    data_x, data_y = utils.preprocess_data(cleaned_version=True, with_filename=False)
    ok_data_x, hold_off_data_x, ok_data_y, hold_off_data_y = train_test_split(data_x, data_y, random_state=10)
    total_data = len(ok_data_x)
    num_train = int(total_data * 0.8)
    num_test = total_data - num_train
    step_size = int(num_train/30)
    
    # proj1 data + more and more external data
    result = []
    for total_train in range(0,num_train,step_size):
        scores = []
        for iteration_count in range(10):
            train_x, test_x, train_y, test_y = train_test_split(ok_data_x, ok_data_y, random_state=iteration_count, train_size=num_train, test_size=num_test)

            x = proj1_data_x + train_x[:total_train]
            y = proj1_data_y + train_y[:total_train]

            # get a brain
            brain = utils.build_a_simple_brain()
            # train
            brain.fit(x, y)
            # accuracy
            score = brain.score(test_x, test_y)
            scores.append(score)
        result.append(scores)
        print(".", end="", flush=True)
    print("")

    # only external-data
    external_scores = []
    for iteration_count in range(10):
        train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, random_state=iteration_count, train_size=num_train, test_size=num_test)

        # get a brain
        brain = utils.build_a_simple_brain()
        # train
        brain.fit(train_x, train_y)
        # accuracy
        score = brain.score(test_x, test_y)
        external_scores.append(score)

    result.append(external_scores)
    plt.figure(figsize=(9,6))
    bplot = plt.boxplot(result)
    bplot['boxes'][-1].set_linestyle('--')
    bplot['boxes'][-1].set_color('red')
    
    plt.xlabel("# of External Training Data Added to Jarvis Brain")
    plt.ylabel("Accuracy Score")
    xtick_val = list(range(0,len(result),4))
    xtick_name = np.array(xtick_val) * step_size
    plt.xticks(xtick_val, xtick_name)
    plt.grid(color='#EEEEEE')
    plt.savefig("adding_external_data.png")
    plt.show()

# # pour all data together and do cross-validation
# def step3():
#     print("Step 3 start. Pour all data together and do cross-validation.")
#     proj1_data_x, proj1_data_y = utils.project1_data()
#     data_x, data_y = utils.preprocess_data(cleaned_version=True, with_filename=False)
#     data_x = data_x + proj1_data_x
#     data_y = data_y + proj1_data_y
#     scores = []
#     for iteration_count in range(20):
#         train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, random_state=iteration_count)
#         # get a brain
#         brain = utils.build_a_simple_brain()
#         # train
#         brain.fit(train_x, train_y)
#         # accuracy
#         score = brain.score(test_x, test_y)
#         scores.append(score)

#     plt.figure(figsize=(9,6))
#     bplot = plt.boxplot(scores)
#     plt.xlabel("")
#     plt.ylabel("Accuracy Score")
#     plt.grid(color='#EEEEEE')
#     plt.savefig("estimate_real_world_test_score.png")
#     plt.show()

if __name__ == "__main__":
    step1()
    step2()
    # step3()