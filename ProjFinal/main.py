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

plot_age_dist(df_2007)
plot_age_dist(df_2017)
