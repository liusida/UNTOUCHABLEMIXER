"""
We want to know:
1. Given the data in 2007, what's the model for PHQ score?
    1.1 make assumption for PHQ score. 
        According to the histogram, we assume PHQ score follows 
        a negative binomial distribution based on the shape (not the logic).
    1.2 use MCMC to generate trace and get the model parameters distribution.
        we see all of them are follow normal-like distribution.
        we pick the mean of the parameter distribution, so we have the expected parameters.
    1.3 Use the expected parameter to generate random data, and compare that with original data.


"""
# %%
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
import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
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
df_2007_c.loc[df_2007_c['SMQ020']==2, 'SMQ040']=3
df_2007_c.loc[df_2007_c['SMQ040'].isna(), 'SMQ040']=9

df_2017_c = df_2017.dropna(subset=['BMXBMI', 'INDFMPIR'])
df_2017_c.loc[df_2017_c['DUQ240']==2, 'DUQ250']=2
df_2017_c.loc[df_2017_c['DUQ240']==2, 'DUQ290']=2
df_2017_c.loc[df_2017_c['DUQ240']==2, 'DUQ330']=2
df_2017_c.loc[df_2017_c['DUQ250'].isna(), 'DUQ250']=9
df_2017_c.loc[df_2017_c['DUQ290'].isna(), 'DUQ290']=9
df_2017_c.loc[df_2017_c['DUQ330'].isna(), 'DUQ330']=9
df_2017_c.loc[df_2017_c['SMQ020']==2, 'SMQ040']=3

print(f"After drop na, shape: {df_2007_c.shape}, {df_2017_c.shape}.")
for cs in picked_columns:
    for c in picked_columns[cs]:
        if np.sum(df_2007_c[c].isna())>0:
            print(f"NaN in 2007 {c}: {np.sum(df_2007_c[c].isna())}")
        if np.sum(df_2017_c[c].isna())>0:
            print(f"NaN in 2017 {c}: {np.sum(df_2017_c[c].isna())}")

df_2007 = df_2007_c
df_2017 = df_2017_c

#%%
def plot_ecdf(datasets, labels, alphas):
    """ Plot several ECDF at once """
    assert len(labels) == len(datasets)
    assert len(alphas) == len(datasets)
    plt.figure(figsize=[9,6])
    for idx, data in enumerate(datasets):
        _plot_ecdf(data, labels[idx], alphas[idx])
    plt.xlabel("PHQ score")
    plt.ylabel("Cumulative Probability")
    plt.legend()
    plt.savefig("saved_plots/ecdf_"+"_".join(labels)+".png")
    plt.show()
def _plot_ecdf(data, label='Value', alpha=1):
    """ Add one ECDF to the plot """
    data = np.array(data)
    data = np.sort(data)
    t = len(data)
    prob = np.arange(t) / t
    plt.plot(data, prob, label=label, alpha=alpha)

def plot_hist(datasets, bins, labels, alphas):
    """ Plot several Histogram at once """
    assert len(labels) == len(datasets)
    assert len(alphas) == len(datasets)
    plt.figure(figsize=[9,6])
    for idx, data in enumerate(datasets):
        plt.hist(data, bins=bins[idx], density=True, label=labels[idx], alpha=alphas[idx])
    plt.xlabel("PHQ score")
    plt.ylabel("Probability")
    plt.legend()
    plt.savefig("saved_plots/hist_"+"_".join(labels)+".png")
    plt.show()

# %%
print(f"histogram for PHQ score in 2007 and 2017")
plot_hist([df_2007['PHQ score'].astype(int), df_2017['PHQ score'].astype(int)], bins=[27,27], labels=["2007","2017"], alphas=[1,0.5])
plot_ecdf([df_2007['PHQ score'].astype(int),df_2017['PHQ score'].astype(int)], labels=["2007","2017"], alphas=[1,0.9])
print(f"According to the plot, we assume a Negative Binomial distribution.")

# %%
print(f"Here is one example of Negative Binomial distribution.")
with pm.Model() as model:
    score_rv = pm.NegativeBinomial('score_rv', mu=3, alpha=0.9)
    x = score_rv.random(size=4000)
plot_hist([x], [27], ["NegativeBinomial\nmu=3, alpha=0.9"], [1])
plot_ecdf([x], ["Random NegativeBinomial\nmu=3, alpha=0.9"], [1])

# %%
def mcmcNegativeBinomial(data):
    """Generate a trace for the data"""
    with pm.Model() as model:
        # Not familiar with Negative Binomial, so no prior knowledge, let's choose uniform as a prior
        # To be safe, make sure the possible range is larger than needed.
        alpha_rv = pm.Uniform('alpha_rv', 0.0, 3.0)
        mu_rv = pm.Uniform('mu_rv', 0.1, 30.0)
        score_rv = pm.NegativeBinomial('score_rv', mu=mu_rv, alpha=alpha_rv, observed=data)
        step = pm.NUTS()
        trace = pm.sample(step=step, draws=10000, chains=4, cores=4, init='adapt_diag')
        graph = pm.model_to_graphviz(model)
    graph.render(filename='model', format='png')
    return trace
# %%
def visualize_trace(trace, data, desc='2007'):
    """Interpret the trace"""
    print(f"Visualize the probability distribution of two parameters.")
    alphas = trace.get_values('alpha_rv')
    mus = trace.get_values('mu_rv')
    fig, ax = plt.subplots(figsize=[9,6], nrows=2)
    ax[0].hist(alphas, bins='auto', density=True)
    ax[0].set_title(f"Probability Distribution of Beta ({desc})")
    ax[1].hist(mus, bins='auto', density=True)
    ax[1].set_title(f"Probability Distribution of q ({desc})")
    plt.tight_layout()
    plt.show()
    print(f"Reconstruct the PHQ score distribution using mean value and compare with original data too see the fitness.")
    mu_mean = np.mean(mus)
    alpha_mean = np.mean(alphas)
    with pm.Model() as model:
        score_rv = pm.NegativeBinomial('score_rv', mu=mu_mean, alpha=alpha_mean)
        x = score_rv.random(size=10000)
    #HACK: I don't know how to bound the model, so what I can do is cut off the tail after getting the data.
    #      However, there's little data that is larger than the boundary, luckily.
    x=x[x<=27]
    plot_ecdf([x,data], labels=[f"Random Data (mu={mu_mean:.3f}, alpha={alpha_mean:.3f})", desc], alphas=[1,0.9])
    plot_hist([x,data], bins=[27,27], labels=[f"Random Data (mu={mu_mean:.3f}, alpha={alpha_mean:.3f})", desc], alphas=[1,0.5])
    return alpha_mean,mu_mean
# %%
print(f"Bayesian Inference: MCMC method on 2007 data")
data = df_2007['PHQ score'].astype(int)
trace = mcmcNegativeBinomial(data)
visualize_trace(trace,data,'2007')

# %%
print(f"What about 2017?")
data = df_2017['PHQ score'].astype(int)
trace = mcmcNegativeBinomial(data)
visualize_trace(trace,data,'2017')

# %%
print(f"Now what about condition on SLQ120: How often feel overly sleepy during day?")
def run_analysis(df, year='2007', variable_name='SLQ120', groups=[3,4], group_desc='Often, Always'):
    df_filter = None
    for v in groups:
        if df_filter is None:
            df_filter = (df[variable_name]==v)
        else:
            df_filter |= (df[variable_name]==v)
    print(f"In {year} data, filtered by {variable_name}=={groups}, {group_desc}")
    data = df[ df_filter ]['PHQ score']
    trace = mcmcNegativeBinomial(data)
    return visualize_trace(trace,data,f'{year} with {variable_name}=={groups}')
def run_analysis_numerical(df, year='2007', variable_name='SLQ120', val_range=[0.0, 1.0], group_desc='Often, Always'):
    df_filter = (df[variable_name]>=val_range[0]) & (df[variable_name]<val_range[1])
    print(f"In {year} data, filtered by {variable_name} in {val_range}, {group_desc}")
    data = df[ df_filter ]['PHQ score']
    trace = mcmcNegativeBinomial(data)
    return visualize_trace(trace,data,f'{year} with {variable_name} in {val_range}')

param = {}
param['2007_Pos'] = run_analysis(df_2007, year='2007', variable_name='SLQ120', groups=[3,4], group_desc="Often, or Always")
param['2007_Neg'] = run_analysis(df_2007, year='2007', variable_name='SLQ120', groups=[0,1,2], group_desc="Never, Rare, or Sometimes")
param['2017_Pos'] = run_analysis(df_2017, year='2017', variable_name='SLQ120', groups=[3,4], group_desc="Often, or Always")
param['2017_Neg'] = run_analysis(df_2017, year='2017', variable_name='SLQ120', groups=[0,1,2], group_desc="Never, Rare, or Sometimes")
print(param)
#%%
print(f"What about SMQ040?")
param = {}
param['2007_Pos'] = run_analysis(df_2007, year='2007', variable_name='SMQ040', groups=[1,2], group_desc="Every day or Sometimes")
param['2007_Neg'] = run_analysis(df_2007, year='2007', variable_name='SMQ040', groups=[3], group_desc="Not at all")
param['2017_Pos'] = run_analysis(df_2017, year='2017', variable_name='SMQ040', groups=[1,2], group_desc="Every day or Sometimes")
param['2017_Neg'] = run_analysis(df_2017, year='2017', variable_name='SMQ040', groups=[3], group_desc="Not at all")
print(param)
# %%
print(f"What about SLQ050? Ever told doctor had trouble sleeping?")
param = {}
param['2007_Pos'] = run_analysis(df_2007, year='2007', variable_name='SLQ050', groups=[1], group_desc="Yes")
param['2007_Neg'] = run_analysis(df_2007, year='2007', variable_name='SLQ050', groups=[2], group_desc="No")
param['2017_Pos'] = run_analysis(df_2017, year='2017', variable_name='SLQ050', groups=[1], group_desc="Yes")
param['2017_Neg'] = run_analysis(df_2017, year='2017', variable_name='SLQ050', groups=[2], group_desc="No")
print(param)

#%%
print(f"PAQ650 - Vigorous recreational activities")
param = {}
param['2007_Pos'] = run_analysis(df_2007, year='2007', variable_name='PAQ650', groups=[1], group_desc="Yes")
param['2007_Neg'] = run_analysis(df_2007, year='2007', variable_name='PAQ650', groups=[2], group_desc="No")
param['2017_Pos'] = run_analysis(df_2017, year='2017', variable_name='PAQ650', groups=[1], group_desc="Yes")
param['2017_Neg'] = run_analysis(df_2017, year='2017', variable_name='PAQ650', groups=[2], group_desc="No")
print(param)

#%%
print(f"PAQ665 - Moderate recreational activities")
param = {}
param['2007_Pos'] = run_analysis(df_2007, year='2007', variable_name='PAQ665', groups=[1], group_desc="Yes")
param['2007_Neg'] = run_analysis(df_2007, year='2007', variable_name='PAQ665', groups=[2], group_desc="No")
param['2017_Pos'] = run_analysis(df_2017, year='2017', variable_name='PAQ665', groups=[1], group_desc="Yes")
param['2017_Neg'] = run_analysis(df_2017, year='2017', variable_name='PAQ665', groups=[2], group_desc="No")
print(param)

#%%
print(f"INDFMPIR - Ratio of family income to poverty")
param = {}
param['2007_Pos'] = run_analysis_numerical(df_2007, year='2007', variable_name='INDFMPIR', val_range=[0.0, 5.0], group_desc="Yes")
param['2007_Neg'] = run_analysis_numerical(df_2007, year='2007', variable_name='INDFMPIR', val_range=[5.0, 6.0], group_desc="No")
param['2017_Pos'] = run_analysis_numerical(df_2017, year='2017', variable_name='INDFMPIR', val_range=[0.0, 5.0], group_desc="Yes")
param['2017_Neg'] = run_analysis_numerical(df_2017, year='2017', variable_name='INDFMPIR', val_range=[5.0, 6.0], group_desc="No")
print(param)

#%%
print(f"MCQ080 - Doctor ever said you were overweight")
param = {}
param['2007_Pos'] = run_analysis(df_2007, year='2007', variable_name='MCQ080', groups=[1], group_desc="Yes")
param['2007_Neg'] = run_analysis(df_2007, year='2007', variable_name='MCQ080', groups=[2], group_desc="No")
param['2017_Pos'] = run_analysis(df_2017, year='2017', variable_name='MCQ080', groups=[1], group_desc="Yes")
param['2017_Neg'] = run_analysis(df_2017, year='2017', variable_name='MCQ080', groups=[2], group_desc="No")
print(param)

#%%
print(f"Obesity BMI>=30, BMXBMI - Body Mass Index (kg/m**2)")
param = {}
param['2007_Pos'] = run_analysis_numerical(df_2007, year='2007', variable_name='BMXBMI', val_range=[0.0, 30.0], group_desc="Yes")
param['2007_Neg'] = run_analysis_numerical(df_2007, year='2007', variable_name='BMXBMI', val_range=[30.0, 9999.0], group_desc="No")
param['2017_Pos'] = run_analysis_numerical(df_2017, year='2017', variable_name='BMXBMI', val_range=[0.0, 30.0], group_desc="Yes")
param['2017_Neg'] = run_analysis_numerical(df_2017, year='2017', variable_name='BMXBMI', val_range=[30.0, 9999.0], group_desc="No")
print(param)

#%%
print(f"DUQ250 - Ever use any form of cocaine")
param = {}
param['2007_Pos'] = run_analysis(df_2007, year='2007', variable_name='DUQ250', groups=[1], group_desc="Yes")
param['2007_Neg'] = run_analysis(df_2007, year='2007', variable_name='DUQ250', groups=[2], group_desc="No")
param['2017_Pos'] = run_analysis(df_2017, year='2017', variable_name='DUQ250', groups=[1], group_desc="Yes")
param['2017_Neg'] = run_analysis(df_2017, year='2017', variable_name='DUQ250', groups=[2], group_desc="No")
print(param)

#%%
print(f"DUQ290 - Ever use any form of cocaine")
param = {}
param['2007_Pos'] = run_analysis(df_2007, year='2007', variable_name='DUQ290', groups=[1], group_desc="Yes")
param['2007_Neg'] = run_analysis(df_2007, year='2007', variable_name='DUQ290', groups=[2], group_desc="No")
param['2017_Pos'] = run_analysis(df_2017, year='2017', variable_name='DUQ290', groups=[1], group_desc="Yes")
param['2017_Neg'] = run_analysis(df_2017, year='2017', variable_name='DUQ290', groups=[2], group_desc="No")
print(param)

#%%
print(f"DUQ330 - Ever use any form of cocaine")
param = {}
param['2007_Pos'] = run_analysis(df_2007, year='2007', variable_name='DUQ330', groups=[1], group_desc="Yes")
param['2007_Neg'] = run_analysis(df_2007, year='2007', variable_name='DUQ330', groups=[2], group_desc="No")
param['2017_Pos'] = run_analysis(df_2017, year='2017', variable_name='DUQ330', groups=[1], group_desc="Yes")
param['2017_Neg'] = run_analysis(df_2017, year='2017', variable_name='DUQ330', groups=[2], group_desc="No")
print(param)


# %%
