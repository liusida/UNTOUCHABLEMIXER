
import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
import step1_preprocess
import theano.tensor as tt
import pymc3.distributions.transforms as tr
import seaborn as sns


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
    

conditions = [
    #(df_2007['RIDAGEYR'] >= 18.0) & (df_2007['RIDAGEYR'] < 20.0),
    (df_2007['RIDAGEYR'] >= 18.0) & (df_2007['RIDAGEYR'] < 40.0),
    (df_2007['RIDAGEYR'] >= 40.0) & (df_2007['RIDAGEYR'] < 60.0),
    (df_2007['RIDAGEYR'] >= 60.0) ]
choices = [1.0, 2.0, 3.0]
df_2007['age_group'] = np.select(conditions, choices, default=0.0)

conditions = [
    #(df_2017['RIDAGEYR'] >= 18.0) & (df_2017['RIDAGEYR'] < 20.0),
    (df_2017['RIDAGEYR'] >= 18.0) & (df_2017['RIDAGEYR'] < 40.0),
    (df_2017['RIDAGEYR'] >= 40.0) & (df_2017['RIDAGEYR'] < 60.0),
    (df_2017['RIDAGEYR'] >= 60.0) ]
choices = [1.0, 2.0, 3.0]
df_2017['age_group'] = np.select(conditions, choices, default=0.0)



    
md_2007 = df_2007[df_2007['Major depression'] == True]
md_2017 = df_2017[df_2017['Major depression'] == True]

y = np.array([md_2007.shape[0], md_2017.shape[0] ])
n = np.array([df_2007.shape[0], df_2017.shape[0]])
N = len(n)

y = df_2007['Major depression'].to_numpy().astype(int)

def mcmcBinomial(data):
    def hyper_prior(value):
        ''' prior density'''
        return tt.log(tt.pow(tt.sum(value), -5/2))


    with pm.Model() as model:
        # Uninformative prior for alpha and beta
        #true_rates = pm.Beta('true_rates', a, b, size=5)
        ab = pm.HalfFlat('alpha_beta',
                     shape=2,
                     testval=np.asarray([1., 1.]))
        pm.Potential('p(alpha, beta)', hyper_prior(ab))

        #Allows you to do algrebra with RVs
        X = pm.Deterministic('X', tt.log(ab[0]/ab[1]))
        Z = pm.Deterministic('Z', tt.log(tt.sum(ab)))

        theta = pm.Beta('theta', alpha=ab[0], beta=ab[1])

        #p = pm.Binomial('y', p=theta, observed=y, n=n)
        p = pm.Bernoulli('y', theta, observed = data)
        trace = pm.sample(5000, tune=2000, target_accept = 0.95)
    return(trace)
    



#%%
def visualize_binomial(trace, data, desc = '2007'):
    sns.kdeplot(trace['X'], trace['Z'], shade = True, cmap = 'viridis')
    plt.xlabel(r'$\log(\alpha/\beta)$', fontsize = 16)
    plt.ylabel(r'$\log(\alpha+\beta)$', fontsize = 16)
    plt.title(r'KDE for $\log(\alpha/\beta)$ and $\log(\alpha+\beta)$', 
              fontsize = 16)
    textstr = '\n'.join(('Mean Values:',
                         r'$\alpha = %.2f$' % (trace['alpha_beta'].mean(axis = 0)[0], ),
                         r'$\beta = %.2f$' % (trace['alpha_beta'].mean(axis = 0)[1], ),
                         ))
    props = dict(boxstyle = 'round', facecolor = 'wheat', alpha = 0.5)
    x_text = max(trace['X']) + 1.5
    y_text = max(trace['Z']) - 10
    plt.text(x_text, y_text, textstr, bbox = props, fontsize = 16)
    plt.show()
    
    sns.kdeplot(trace['alpha_beta'][:,0], trace['alpha_beta'][:,1], 
                shade = True, cmap = 'viridis')
    plt.xlabel(r'$\alpha$', fontsize = 16)
    plt.ylabel(r'$\beta$', fontsize = 16)
    plt.title(r'KDE for $\alpha$ and $\beta$', 
              fontsize = 16)
    plt.show()
    
    
def visualize_compare(trace_list = None, desc_list = ['2007','2017'], 
                      params = 'theta'):
    #params1 = trace1.get_values(params)
    #params2 = trace2.get_values(params)
    
    for i in range(len(trace_list)):
        p = trace_list[i].get_values(params)
        sns.kdeplot(p, label = desc_list[i])
    #sns.kdeplot(params2, label = desc2)
    plt.legend()
    #plt.show()

# %%
print(f"Bayesian Inference: MCMC method on 2007 data")
data = df_2007['Major depression'].astype(int)
trace1 = mcmcBinomial(data)
visualize_binomial(trace1,data,'2007')

# %%
print(f"What about 2017?")
data = df_2017['Major depression'].astype(int)
trace2 = mcmcBinomial(data)
visualize_binomial(trace2,data,'2017')

visualize_compare(trace_list = [trace1, trace2])

# %%
print(f"Now what about condition on SLQ120: How often feel overly sleepy during day?")
def run_analysis(df, year='2007', variable_name='SLQ120', 
                 groups=[3,4], group_desc='Often, Always'):
    df_filter = None
    for v in groups:
        if df_filter is None:
            df_filter = (df[variable_name]==v)
        else:
            df_filter |= (df[variable_name]==v)
    print(f"In {year} data, filtered by {variable_name}=={groups}, {group_desc}")
    data = df[ df_filter ]['Major depression']
    trace = mcmcBinomial(data)
    return trace, visualize_binomial(trace,data,f'{year} with {variable_name}=={groups}')

param = {}
param['2007_34'] = run_analysis(df_2007, year='2007', variable_name='SLQ120', 
     groups=[3,4], group_desc="Often, or Always")
param['2007_012'] = run_analysis(df_2007, year='2007', variable_name='SLQ120', 
     groups=[0,1,2], group_desc="Never, Rare, or Sometimes")
param['2017_34'] = run_analysis(df_2017, year='2017', variable_name='SLQ120', 
     groups=[3,4], group_desc="Often, or Always")
param['2017_012'] = run_analysis(df_2017, year='2017', variable_name='SLQ120', 
     groups=[0,1,2], group_desc="Never, Rare, or Sometimes")

visualize_compare(trace_list = [param['2007_34'][0],
                                param['2007_012'][0],
                                param['2017_34'][0],
                                param['2017_012'][0]], desc_list = ['2007 Often Sleepy',
                                     '2007 Rarely Sleepy',
                                     '2017 Often Sleepy',
                                     '2017 Rarely Sleepy'])
plt.title('KDE of Proportion with Major Depression for SLQ120: \n How often do you feel overly sleepy during the day?')


param['2007_male'] = run_analysis(df_2007, year='2007', variable_name='RIAGENDR', 
     groups=[1], group_desc="Male")
param['2007_female'] = run_analysis(df_2007, year='2007', variable_name='RIAGENDR', 
     groups=[2], group_desc="Female")
param['2017_male'] = run_analysis(df_2017, year='2017', variable_name='RIAGENDR', 
     groups=[1], group_desc="Male")
param['2017_female'] = run_analysis(df_2017, year='2017', variable_name='RIAGENDR', 
     groups=[2], group_desc="Female")

visualize_compare(trace_list = [param['2007_male'][0],
                                param['2007_female'][0],
                                param['2017_male'][0],
                                param['2017_female'][0]], desc_list = ['2007 Male',
                                     '2007 Female',
                                     '2017 Male',
                                     '2017 Female'])
    
param['2007_age1'] = run_analysis(df_2007, year='2007', variable_name='age_group', 
     groups=[1], group_desc="18-39")
param['2007_age2'] = run_analysis(df_2007, year='2007', variable_name='age_group', 
     groups=[2], group_desc="40-59")
param['2007_age3'] = run_analysis(df_2007, year='2007', variable_name='age_group', 
     groups=[3], group_desc="60+")
param['2017_age1'] = run_analysis(df_2017, year='2017', variable_name='age_group', 
     groups=[1], group_desc="18-39")
param['2017_age2'] = run_analysis(df_2017, year='2017', variable_name='age_group', 
     groups=[2], group_desc="40-59")
param['2017_age3'] = run_analysis(df_2017, year='2017', variable_name='age_group', 
     groups=[3], group_desc="60+")


visualize_compare(trace_list = [param['2007_age1'][0],
                                param['2007_age2'][0],
                                param['2007_age3'][0],
                                param['2017_age1'][0],
                                param['2017_age2'][0],
                                param['2017_age3'][0]], desc_list = ['2007 Ages 18-39',
                                     '2007 Ages 40 - 59',
                                     '2007 Ages 60+',
                                     '2017 Ages 18 - 39',
                                     '2017 Ages 40 - 59',
                                     '2017 60+'])
    
#pm.traceplot(trace, var_names = ['ab','X','Z'])
#sns.kdeplot(trace['X'], trace['Z'], shade = True, cmap = 'viridis')

#pm.plot_posterior(trace, var_names = ['theta'])

#trace['ab'].mean(axis = 0)
