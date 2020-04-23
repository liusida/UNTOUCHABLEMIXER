import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import step1_preprocess

# Read data
if os.path.exists("cache/df_2007.pickle"):
    df_2007 = pd.read_pickle("cache/df_2007.pickle")
else:
    df_2007 = step1_preprocess.load_full_data(2007)
    df_2007.to_pickle("cache/df_2007.pickle")

if os.path.exists("cache/df_2017.pickle"):
    df_2017 = pd.read_pickle("cache/df_2017.pickle")
else:
    df_2017 = step1_preprocess.load_full_data(2017)
    df_2017.to_pickle("cache/df_2017.pickle")

def plot_age_dist(df_big):
    plt.hist(df_big[df_big['Major depression']==True]['RIDAGEYR'], bins='auto', alpha=0.2, color='red')
    plt.hist(df_big[df_big['Major depression']==False]['RIDAGEYR'], bins='auto', alpha=0.2, color='blue')
    plt.show()

# plot_age_dist(df_2007)
# plot_age_dist(df_2017)
def match_columns(df_2007, df_2017):
    """ print out the matching columns """
    c17 = df_2017.columns
    c07 = df_2007.columns
    count_both = 0
    count_only_17 = 0
    count_only_07 = 0
    for c in c17:
        if c in c07:
            count_both+=1
        else:
            count_only_17+=1
    for c in c07:
        if c not in c17:
            count_only_07+=1
    print(f"There are {df_2007.shape[1]} features in 2007 dataset.")
    print(f"There are {df_2017.shape[1]} features in 2017 dataset.")
    print(f"{count_both} features are in both dataset; {count_only_17} only in 07 but not 17 , {count_only_07} only in 17 not in 07.")
    for c in c17:
        if c in c07:
            print(c, end=', ')

# match_columns(df_2007, df_2017)