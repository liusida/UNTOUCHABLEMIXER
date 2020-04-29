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
Beta = st.beta.pdf
import step1_preprocess

#%%
# Constants
colors = {
    2007: "#E0162B",
    2017: "#0052A5"
}
x = np.linspace(0.0,0.17,300)

#%%
# Helper Functions
def get_counts(df, col, groups):
    """Count Depression and Not-depression, respectively"""
    MajorDepression = {}
    NotMajorDepression = {}
    for val in groups:
        MajorDepression[val] = ((df[col]==val) & (df['Major depression']==True)).sum()
        NotMajorDepression[val] = ((df[col]==val) & (df['Major depression']==False)).sum()
    return MajorDepression, NotMajorDepression

def analyze(title, col, groups, group_names, line_styles, x = np.linspace(0.0,0.08,100)):
    """compute Beta posterior and plot"""
    depression = {}
    not_depression = {}
    Posterior = {}
    plt.figure(figsize=[9,2.5])
    for year in [2007,2017]:
        depression[year], not_depression[year] = get_counts(df[year], col, groups)

        print(f"== {year} Data ==")
        print(f"\t\tDepression\tNon-depression")
        for g in groups:
            print(f"{group_names[g]}\t\t{depression[year][g]}\t\t{not_depression[year][g]}")

        Posterior[year] = {}
        for g in groups:
            Posterior[year][g] = Beta(x, depression[year][g] +1, not_depression[year][g] +1 )

        for g in groups:
            plt.plot(x, Posterior[year][g], line_styles[g], label=f'{group_names[g]} {year}', c=colors[year])
            print(f"{year} {group_names[g]} peaks at {x[np.argmax(Posterior[year][g])]}")
    plt.title(title)
    plt.legend()
    plt.xlabel(r"$\theta$")
    plt.ylabel("Probability Density")
    plt.tight_layout()
    plt.savefig(f"saved_plots/{title}.png")
    plt.show()

#%%
print(f"Reading data.")
df = {}
if os.path.exists("cache/df_2007.pickle"):
    df[2007] = pd.read_pickle("cache/df_2007.pickle")
else:
    df[2007] = step1_preprocess.load_full_data(2007)
    df[2007].to_pickle("cache/df_2007.pickle")

if os.path.exists("cache/df_2017.pickle"):
    df[2017] = pd.read_pickle("cache/df_2017.pickle")
else:
    df[2017] = step1_preprocess.load_full_data(2017)
    df[2017].to_pickle("cache/df_2017.pickle")

# %%
title = 'Overall'
for year in [2007,2017]:
    df[year]['Overall'] = 1

col = 'Overall'
groups = [1]
group_names = {1:""}
line_styles = {1:"-"}
# x = np.linspace(0.01,0.075,100)
analyze(title, col, groups, group_names, line_styles, x)

# %%
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
title = 'Gender'
col = 'RIAGENDR'
groups = [1,2]
group_names = {1:"Male", 2:"Female"}
line_styles = {1:"-", 2:":"}
# x = np.linspace(0.01,0.075,100)
analyze(title, col, groups, group_names, line_styles, x)


# %%
"""
2. The difference in proportion of Major depression between 2007, 2017, different age groups.
    Formula:
    Posterior = Beta(a+1, b+1)
    Memo:
    age_group==1 for 18-40
    age_group==2 for 40-60
    age_group==3 for >60
"""
title = 'Age'
for year in [2007,2017]:
    conditions = [
        (df[year]['RIDAGEYR'] >= 18.0) & (df[year]['RIDAGEYR'] < 40.0),
        (df[year]['RIDAGEYR'] >= 40.0) & (df[year]['RIDAGEYR'] < 60.0),
        (df[year]['RIDAGEYR'] >= 60.0) ]
    choices = [1,2,3]
    df[year]['age_group'] = np.select(conditions, choices, default=0.0)

col = 'age_group'
groups = [1,2,3]
group_names = {1:"18-40", 2:"40-60", 3:">60"}
line_styles = {1:"-.", 2:"-", 3:":"}
# x = np.linspace(0.01,0.085,100)

analyze(title, col, groups, group_names, line_styles, x)


# %%
title = 'SLQ120 - How often feel overly sleepy during day?'
for year in [2007,2017]:
    conditions = [
        (df[year]['SLQ120'] >= 0.0) & (df[year]['SLQ120'] < 2.1),
        (df[year]['SLQ120'] >= 2.1) & (df[year]['SLQ120'] < 4.1)]
    choices = [1,2]
    df[year]['sleepy_group'] = np.select(conditions, choices, default=0.0)

col = 'sleepy_group'
groups = [1,2]
group_names = {1:"Not Sleepy", 2:"Often Sleepy"}
line_styles = {1:"-.", 2:"-"}
# x = np.linspace(0.0,0.17,100)

analyze(title, col, groups, group_names, line_styles, x)


# %%
title = 'SLQ050 - Ever told doctor had trouble sleeping?'
col = 'SLQ050'
groups = [1,2]
group_names = {1:"Yes", 2:"No"}
line_styles = {1:"-", 2:"-."}
# x = np.linspace(0.0,0.17,100)

analyze(title, col, groups, group_names, line_styles, x)


# %%
title = 'SMQ020 - Smoked at least 100 cigarettes in life'
for year in [2007,2017]:
    conditions = [
        (df[year]['SMQ020' ]== 2.0),
        (df[year]['SMQ020'] == 1.0) & (df[year]['SMQ040'] <= 2.0),
        (df[year]['SMQ020'] == 1.0) & (df[year]['SMQ040'] > 2.0) ]
    choices = [1,2,3]
    df[year]['smoking'] = np.select(conditions, choices, default=0.0)
title = 'Smoking Status'
col = 'smoking'
groups = [1,2,3]
group_names = {1:"Never", 2:"Current", 3:"Past"}
line_styles = {1:"-.", 2:"-", 3:":"}
# x = np.linspace(0.0,0.08,100)

analyze(title, col, groups, group_names, line_styles, x)

# %%
title = 'PAQ650 - Vigorous recreational activities'
col = 'PAQ650'
groups = [1,2]
group_names = {1:"Yes", 2:"No"}
line_styles = {1:"-", 2:"-."}
# x = np.linspace(0.0,0.07,100)

analyze(title, col, groups, group_names, line_styles, x)

# %%
title = 'PAQ665 - Moderate recreational activities'
col = 'PAQ665'
groups = [1,2]
group_names = {1:"Yes", 2:"No"}
line_styles = {1:"-", 2:"-."}
# x = np.linspace(0.0,0.07,100)

analyze(title, col, groups, group_names, line_styles, x)

# %%
title = 'INDFMPIR - Ratio of family income to poverty'
for year in [2007,2017]:
    conditions = [
        (df[year]['INDFMPIR'] >= 0.0) & (df[year]['INDFMPIR'] < 1.3),
        (df[year]['INDFMPIR'] >= 1.3) & (df[year]['INDFMPIR'] < 2.5),
        (df[year]['INDFMPIR'] >= 2.5) & (df[year]['INDFMPIR'] < 5.1),
        (df[year]['INDFMPIR'].isna()),
        ]
    choices = [1,2,3,4]
    df[year]['income_group'] = np.select(conditions, choices, default=0.0)

col = 'income_group'
groups = [1,2,3]
group_names = {1:"Low", 2:"Mid",3:"High",4:"Missing"}
line_styles = {1:"-.", 2:"--",3:'-',4:":"}
# x = np.linspace(0.0,0.09,100)

analyze(title, col, groups, group_names, line_styles, x)

# %%
title = 'MCQ080 - Doctor ever said you were overweight'
col = 'MCQ080'
groups = [1,2]
group_names = {1:"Yes", 2:"No"}
line_styles = {1:"-", 2:"-."}
# x = np.linspace(0.0,0.085,100)

analyze(title, col, groups, group_names, line_styles, x)


# %%
title = 'BMXBMI - Body Mass Index'
for year in [2007,2017]:
    conditions = [
        (df[year]['BMXBMI'] >= 0.0) & (df[year]['BMXBMI'] < 30.0),
        (df[year]['BMXBMI'] >= 30.0) & (df[year]['BMXBMI'] < 99.1),
        (df[year]['BMXBMI'].isna()),
        ]
    choices = [1,2,3]
    df[year]['weight_group'] = np.select(conditions, choices, default=0.0)
col = 'weight_group'
groups = [1,2]
group_names = {1:"Normal", 2:"Overweight", 3:"Missing"}
line_styles = {1:"-", 2:"-.", 3:":"}
# x = np.linspace(0.0,0.085,100)

analyze(title, col, groups, group_names, line_styles, x)

# %%
title = 'DUQ240 - Ever used cocaine heroin or methamphetamine'
col = 'DUQ240'
groups = [1,2]
group_names = {1:"Yes", 2:"No"}
line_styles = {1:"-", 2:"-."}
# x = np.linspace(0.0,0.13,100)

analyze(title, col, groups, group_names, line_styles, x)

# %%
title = 'DUQ200 - Ever used marijuana or hashish'
col = 'DUQ200'
groups = [1,2]
group_names = {1:"Yes", 2:"No"}
line_styles = {1:"-", 2:"-."}
# x = np.linspace(0.0,0.13,100)

analyze(title, col, groups, group_names, line_styles, x)


# %%
