"""
Conclusion:
    We used this file to pick several interesting variables from calcluating the correlation.
"""
#%%
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
    """ plot the distribution relationship between age and depression """
    plt.hist(df_big[df_big['Major depression']==True]['RIDAGEYR'], bins='auto', alpha=0.2, color='red')
    plt.hist(df_big[df_big['Major depression']==False]['RIDAGEYR'], bins='auto', alpha=0.2, color='blue')
    plt.show()

# plot_age_dist(df_2007)
# plot_age_dist(df_2017)
def match_columns(df_2007, df_2017):
    """ return the matching columns """
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
    columns = []
    for c in c17:
        if c in c07:
            # print(c, end=', ')
            columns.append(c)
    return columns

columns = match_columns(df_2007, df_2017)
df_2007_small = df_2007[columns]
df_2017_small = df_2017[columns]

# %%
print("Simply fill NaN with number 9, to avoid corr() function exclude them, otherwise the one with a lot of NaN may end up with a very high score.")
df_2007_small = df_2007_small.fillna(9)
print("Compute pairwised correlation.")
corr_map = df_2007_small.corr()
plt.imshow(corr_map)
plt.colorbar()

# %%
print("sort the columns by correlation with PHQ score in descending order, use absolut value since negative correlation is also interesting.")
related_to_phq = corr_map['PHQ score'].abs()
hi_corr_columns = related_to_phq.sort_values(ascending=False)
hi_corr_columns.index

# %%
print("Produce the pairwised correlation table again, in sorted order. A little bit slow.")
df_2007_small_sorted = df_2007_small[hi_corr_columns.index]
sorted_corr_map = df_2007_small_sorted.corr()
sorted_corr_map.head(40)
#%%
print("The most related columns is itself, Major depression, and DPQ scores, of couse. We only need the rest columns.")
related_cols = []
for c in sorted_corr_map.index.tolist()[:50]:
    if c!= 'PHQ score' and c[:3]!='DPQ' and c!= 'Major depression':
        related_cols.append(c)
related_cols

#%%
plt.imshow(sorted_corr_map)
plt.colorbar()

# %%
print("Save PHQ score and SLQ120 as a csv file.")
sleepy = df_2007_small[['PHQ score','SLQ120']].astype(int).sort_values(by='PHQ score', ascending=False)
sleepy.astype(int).reset_index(drop=True).to_csv("sleepy.csv")


# %%
print("The correlation of columns that's from Katherine's list.")
kw = ['RIAGENDR','RIDAGEYR','RIDRETH1','DMQMILIZ','DMDEDUC3','DMDEDUC2','DMDSCHOL','DMDMARTL','INDHHIN2','INDFMIN2','INDFMPIR','DMDHHSIZ','DMDFMXIZ','BMXWT','BMIHT','BMXBMI','BMXWAIST','BPXPLS','BPXPULS','BPXSY1','BPXDI1','LBXTC','DUQ200','DUQ210','DUQ230','DUQ250','DUQ260','DUQ280','DUQ290','DUQ300','DUQ320','DUQ330','DUQ340','DUQ360','HUQ090','HUQ010','HUQ020','HUQ050','HUQ071','HUD080','MCQ010','MCQ025','MCQ035','MCQ053','MCQ080','MCQ160a','MCQ160c','MCQ220','MCQ170M','MCQ160K','MCQ160E','MCQ160F','MCQ160L','MCQ160B','MCQ300a','OCQ210','OCQ260','OCD395','OCQ180','OCQ670','OCQ265','PAQ650','PAQ655','PAD660','PAQ665','PAQ670','PAD675','PAD680','PAQ605','PAQ635','PAQ610','PAD615','PAQ640','PAD645','PFQ059','PFQ049','SLD012','SLD010H','SMQ040','SMQ020','BPQ020','BPQ080']
for k in kw:
    if k in sorted_corr_map['PHQ score']:
        print(sorted_corr_map['PHQ score'][k])
    else:
        print("")

# %%
