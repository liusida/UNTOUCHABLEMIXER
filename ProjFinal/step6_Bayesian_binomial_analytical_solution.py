"""
Compute posterior using analytical method instead of PyMC3.
"""
#%%
import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
import scipy.stats as st
import step1_preprocess

print(f"Read data.")
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

"""
1. The difference in proportion of Major depression between 2007, 2017, Female, Male.
    Formula:
    Posterior_Male_2007 = Beta( MajorDepression_Male_2007 +1, NotMajorDepression_Male_2007 +1 )
    Posterior_Female_2007 = Beta( MajorDepression_Female_2007 +1, NotMajorDepression_Female_2007 +1 )
    Posterior_Male_2017 = Beta( MajorDepression_Male_2017 +1, NotMajorDepression_Male_2017 +1 )
    Posterior_Female_2017 = Beta( MajorDepression_Female_2017 +1, NotMajorDepression_Female_2017 +1 )
    Memo:
    RIAGENDR==1 for Male
    RIAGENDR==2 for Female
"""

MajorDepression_Male_2007 = ((df_2007['RIAGENDR']==1) & (df_2007['Major depression']==True)).sum()
NotMajorDepression_Male_2007 = ((df_2007['RIAGENDR']==1) & (df_2007['Major depression']==False)).sum()
MajorDepression_Female_2007 = ((df_2007['RIAGENDR']==2) & (df_2007['Major depression']==True)).sum()
NotMajorDepression_Female_2007 = ((df_2007['RIAGENDR']==2) & (df_2007['Major depression']==False)).sum()

MajorDepression_Male_2017 = ((df_2017['RIAGENDR']==1) & (df_2017['Major depression']==True)).sum()
NotMajorDepression_Male_2017 = ((df_2017['RIAGENDR']==1) & (df_2017['Major depression']==False)).sum()
MajorDepression_Female_2017 = ((df_2017['RIAGENDR']==2) & (df_2017['Major depression']==True)).sum()
NotMajorDepression_Female_2017 = ((df_2017['RIAGENDR']==2) & (df_2017['Major depression']==False)).sum()

print(f"== 2007 Data ==")
print(f"\t\tDepression\tNon-depression")
print(f"Male\t\t{MajorDepression_Male_2007}\t\t{NotMajorDepression_Male_2007}")
print(f"Female\t\t{MajorDepression_Female_2007}\t\t{NotMajorDepression_Female_2007}")
print(f"== 2017 Data ==")
print(f"\t\tDepression\tNon-depression")
print(f"Male\t\t{MajorDepression_Male_2017}\t\t{NotMajorDepression_Male_2017}")
print(f"Female\t\t{MajorDepression_Female_2017}\t\t{NotMajorDepression_Female_2017}")

x = np.linspace(0.0,0.08,100)
Beta = st.beta.pdf
Posterior_Male_2007 = Beta(x, MajorDepression_Male_2007 +1, NotMajorDepression_Male_2007 +1)
Posterior_Female_2007 = Beta(x, MajorDepression_Female_2007 +1, NotMajorDepression_Female_2007 +1 )
Posterior_Male_2017 = Beta(x, MajorDepression_Male_2017 +1, NotMajorDepression_Male_2017 +1 )
Posterior_Female_2017 = Beta(x, MajorDepression_Female_2017 +1, NotMajorDepression_Female_2017 +1 )
plt.plot(x, Posterior_Male_2007, label='Male_2007')
plt.plot(x, Posterior_Female_2007, '--', label='Female_2007')
plt.plot(x, Posterior_Male_2017, label='Male_2017')
plt.plot(x, Posterior_Female_2017, '--', label='Female_2017')
plt.legend()
plt.show()


# %%
