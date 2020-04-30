#%%
"""
In this file, we are going to pick six groups of variables and build predictive models out of them
For y labels, we are going to use the boolean "Major depression" column

Conclusion:
    our predictive model didn't work out in this case.
    even combine all columns we picked, the model still tends to say no one is having depression.
    We move on to step4, using Bayesian inference to see the relationship between those variables and depression.
"""
import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import step1_preprocess
import scipy.stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

"""
The six groups are:
1. Sleep
2. Smoke
3. Sport
4. Income
5. Weight
6. Drug Use
"""
picked_columns = {
    'sleep' : ['SLQ120', 'SLQ050'],
    'smoke' : ['SMQ040'],
    'sport' : ['PAQ650', 'PAQ665'],
    'income' : ['INDFMPIR'],
    'weight' : ['MCQ080', 'BMXBMI'],
    'drug' : ['DUQ250', 'DUQ290', 'DUQ330'],
}
all_columns = []
for cs in picked_columns:
    for c in picked_columns[cs]:
        all_columns.append(c)

#%%
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

df_2007.shape
#%%
print(f"The proportion of major depression in both years:")

print(f"(2007) {np.sum(df_2007['Major depression'])} / {df_2007.shape[0]} = {np.sum(df_2007['Major depression']) / df_2007.shape[0]} ")
print(f"(2017) {np.sum(df_2017['Major depression'])} / {df_2017.shape[0]} = {np.sum(df_2017['Major depression']) / df_2017.shape[0]} ")

# %%
print(f"Check all the picked columns are in two datasets.")
for cs in picked_columns:
    for c in picked_columns[cs]:
        if c not in df_2007 or c not in df_2017:
            print(f"No column {c}.")
        if np.sum(df_2007[c].isna())>0:
            print(f"NaN in 2007 {c}: {np.sum(df_2007[c].isna())}")
        if np.sum(df_2017[c].isna())>0:
            print(f"NaN in 2017 {c}: {np.sum(df_2017[c].isna())}")
"""
There isn't any NaN in most columns.

There are a small amount of missing data in BMXBMI and INDFMPIR, we can simply drop records with NaN.

There are many NaN related to Smoke and Drug use. 
We use DUQ240 and SMQ020 to determine how to fill NaN.
The majority case is DUQ240 or SMQ020 is 2, fill NaN with 2, which means No.
If DUQ240 or SMQ020 is 7 or 9 or NaN, drop those records, which means Refused or Don't know.
we treat NaN as No, since it is answered in DUQ240.
"""
df_2007_c = df_2007.dropna(subset=['BMXBMI', 'INDFMPIR'])
df_2007_c.loc[df_2007_c['DUQ240']==2, 'DUQ250']=2
df_2007_c.loc[df_2007_c['DUQ240']==2, 'DUQ290']=2
df_2007_c.loc[df_2007_c['DUQ240']==2, 'DUQ330']=2
df_2007_c.loc[df_2007_c['DUQ250'].isna(), 'DUQ250']=9
df_2007_c.loc[df_2007_c['DUQ290'].isna(), 'DUQ290']=9
df_2007_c.loc[df_2007_c['DUQ330'].isna(), 'DUQ330']=9
df_2007_c.loc[df_2007_c['SMQ020']==2, 'SMQ040']=2
df_2007_c.loc[df_2007_c['SMQ040'].isna(), 'SMQ040']=9

df_2017_c = df_2017.dropna(subset=['BMXBMI', 'INDFMPIR'])
df_2017_c.loc[df_2017_c['DUQ240']==2, 'DUQ250']=2
df_2017_c.loc[df_2017_c['DUQ240']==2, 'DUQ290']=2
df_2017_c.loc[df_2017_c['DUQ240']==2, 'DUQ330']=2
df_2017_c.loc[df_2017_c['DUQ250'].isna(), 'DUQ250']=9
df_2017_c.loc[df_2017_c['DUQ290'].isna(), 'DUQ290']=9
df_2017_c.loc[df_2017_c['DUQ330'].isna(), 'DUQ330']=9
df_2017_c.loc[df_2017_c['SMQ020']==2, 'SMQ040']=2

print(f"After drop na, shape: {df_2007_c.shape}, {df_2017_c.shape}.")
for cs in picked_columns:
    for c in picked_columns[cs]:
        if np.sum(df_2007_c[c].isna())>0:
            print(f"NaN in 2007 {c}: {np.sum(df_2007_c[c].isna())}")
        if np.sum(df_2017_c[c].isna())>0:
            print(f"NaN in 2017 {c}: {np.sum(df_2017_c[c].isna())}")
#%%

# %%
print(f"Preparing data y.")
data_y = df_2017_c['Major depression']
# data_y = df_2017_c['PHQ score']
data_y = data_y.to_numpy().astype(float).reshape(-1,1)
# %%
data_x = df_2017_c[all_columns]
data_x = data_x.to_numpy()

#%%
train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, random_state=10)

# %%
clf = LogisticRegression().fit(train_x,train_y)

# %%
clf.score(test_x, test_y)
# %%
np.max(clf.predict(test_x))

# %%
np.max(test_y)

# %%
np.sum(clf.predict(test_x))

# %%
df = df_2017_c[all_columns]
from factor_analyzer import FactorAnalyzer
fa = FactorAnalyzer()
fa.fit(df)
ev, v = fa.get_eigenvalues()
ev
# %%
plt.scatter(range(1,df.shape[1]+1),ev)
plt.plot(range(1,df.shape[1]+1),ev)
plt.title('Scree Plot')
plt.xlabel('Factors')
plt.ylabel('Eigenvalue')
plt.grid()
plt.show()
# %%
fa = FactorAnalyzer()
fa.analyze(df, 3, rotation="varimax")

#%%
plt.hist(df_2017_c['PHQ score'], bins='auto')
# %%
"""Shot, Dead End, Try Bayesian Inference."""

"""
For example, there's smoke or not. and for smoker, the prob of depression is less than 4%, for non-smoker, less than 4% as well (edited) 
so, no matter what x input, it will be better to guess NO.
"""