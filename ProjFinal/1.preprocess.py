"""
This file preprocess the data from National Health and Nutrition Examination Survey (NHANES) 2007-2008 and 2017-2018
 in the data/ folder into one variable.
 
The NHANES data is in XPT format, and can be read by pandas using API pd.read_sas.

The final usable variable will be saved in a pickel file.

More about the XPT files:
the suffix "_J" stands for 2017-2018, the suffix "_E" stands fro 2007-2008

"""
#%%
import pandas as pd
import numpy as np
def read_sas(filename):
    df_y = pd.read_sas(filename)
    """ Clean step 2: use dtype=int for SEQN"""
    df_y['SEQN'] = df_y['SEQN'].astype(int)
    return df_y

def preprocess(df_y, with_debug_info=True):
    """ 
    The diagnosis of Major depression and the PHQ score from 0-27, refer to
    https://onlinelibrary.wiley.com/doi/10.1046/j.1525-1497.2001.016009606.x """

    if with_debug_info:
        print(f"Shape before preprocess: {df_y.shape}")
    """ Clean step 2: drop answers contain "Refused" and "Don't know" """
    drop_subset = []
    for i in range(9):
        df_y[df_y[f"DPQ0{i+1}0"]>3] = np.nan
    df_y = df_y.dropna(how="all")
    if with_debug_info:
        print(f"Shape after dropping refused and don't know: {df_y.shape}")
    """ Clean step 3: drop NaN is 9 questions. note: DPQ100 is ok to be NaN. """
    for i in range(9):
        drop_subset.append(f"DPQ0{i+1}0")
    df_y = df_y.dropna(subset=drop_subset)
    if with_debug_info:
        print(f"Shape after dropping not finished: {df_y.shape}")
    """ Clean step 4: mark missing data in DPQ100 as 8, so that we can work in integers """
    df_y = df_y.fillna(value={'DPQ100':8})
    df_y = df_y.astype(int)

    # In the paper, the author wrote,
    # "As a severity measure, the PHQ-9 score can range from 0 to 27, since each of the 9 items can be scored from 0 (not at all) to 3 (nearly every day)."
    df_y['PHQ score'] = (df_y['DPQ010'] + df_y['DPQ020'] + df_y['DPQ030'] + df_y['DPQ040'] + df_y['DPQ050'] + df_y['DPQ060'] + df_y['DPQ070'] + df_y['DPQ080'] + df_y['DPQ090']).astype(int)
    # "Major depression is diagnosed if 5 or more of the 9 depressive symptom criteria have been present at least ``more than half the days'' in the past 2 weeks, and 1 of the symptoms is depressed mood or anhedonia."
    df_y['Major depression'] = \
        (((df_y['DPQ010']>=2).astype(int) + (df_y['DPQ020']>=2).astype(int) + (df_y['DPQ030']>=2).astype(int) + (df_y['DPQ040']>=2).astype(int) + (df_y['DPQ050']>=2).astype(int) + (df_y['DPQ060']>=2).astype(int) + (df_y['DPQ070']>=2).astype(int) + (df_y['DPQ080']>=2).astype(int) + (df_y['DPQ090']>=2).astype(int)) >=5 ) & \
        ((df_y['DPQ010']>=2) | (df_y['DPQ020']>=2))

    if with_debug_info:
        print(f"Shape after calculate the major depression and PHQ score: {df_y.shape}")
        print(f"# of Major depression: {np.sum(df_y['Major depression']==True)}/{df_y.shape[0]}")
    return df_y

if __name__ == "__main__":
    """ below are usage examples:
    1. read sas, 2. preprocess, 3. merge with x data, 4. plot."""

    # read sas and preprocess
    df_y = read_sas("data/2007-2008/Questionnaire Data/DPQ_E.XPT")
    # df_y_17 = read_sas("data/2017-2018/Questionnaire Data/DPQ_J.XPT")
    df_y = preprocess(df_y)
    # df_y_17 = preprocess(df_y_17)

    # merge with x data
    filename = "data/2007-2008/DEMO_E.XPT"
    df_x = pd.read_sas(filename)
    df_big = pd.merge(df_x,df_y,on='SEQN')

    df_big
    # plot
    import matplotlib.pyplot as plt
    plt.hist(df_big[df_big['Major depression']==True]['RIDAGEYR'], bins='auto', alpha=0.2, color='red')
    plt.hist(df_big[df_big['Major depression']==False]['RIDAGEYR'], bins='auto', alpha=0.2, color='blue')
    plt.show()



# %%
