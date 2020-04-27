"""
We want to know:
1. Given the data in 2007, what's the model for PHQ score?
    1.1 make assumption for PHQ score. 
        According to the histogram, we assume PHQ score follows 
        an exponential distribution with parameter alpha.

"""
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
    plt.savefig("ecdf_"+"_".join(labels)+".png")
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
    plt.savefig("hist_"+"_".join(labels)+".png")
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

#%%
print(f"Here is one example of Negative Binomial distribution.")
with pm.Model() as model:
    score_rv = pm.Binomial('score_rv', n=27, p=0.06)
    x = score_rv.random(size=4000)
plot_hist([x], [27], [""], [1])

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
        trace = pm.sample(step=step, draws=4000, chains=4, cores=4, init='adapt_diag')
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
        x = score_rv.random(size=4000)
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

param = {}
param['2007_34'] = run_analysis(df_2007, year='2007', variable_name='SLQ120', groups=[3,4], group_desc="Often, or Always")
param['2007_012'] = run_analysis(df_2007, year='2007', variable_name='SLQ120', groups=[0,1,2], group_desc="Never, Rare, or Sometimes")
param['2017_34'] = run_analysis(df_2017, year='2017', variable_name='SLQ120', groups=[3,4], group_desc="Often, or Always")
param['2017_012'] = run_analysis(df_2017, year='2017', variable_name='SLQ120', groups=[0,1,2], group_desc="Never, Rare, or Sometimes")

#%%
param

# %%

# %%
