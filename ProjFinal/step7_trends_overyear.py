"""
Conclusion:
    We used this file to plot the trends, comparing data from NHANES and Google Trends
"""
#%%
import matplotlib.pyplot as plt
import pandas as pd

import step1_preprocess

#%%
# Google Trends
# csv file downloaded from: https://trends.google.com/trends/explore?date=2004-01-01%202020-04-29&geo=US&q=depression

filename = 'data/timeline/multiTimeline.csv'
with open(filename, 'r') as f:
    lines = f.readlines()

m = []
v = []
for i in range(len(lines)):
    if i < 3:
        continue
    s = lines[i].split(',')
    m.append(s[0])
    v.append(int(s[1]))
df = pd.DataFrame([m,v]).T
df['month'] = df[0].str[-2:].astype(int)
df['year'] = df[0].str[:4].astype(int)

# %%
# Plot google trends based on months
v_monthly = []
for i in range(1,13,1):
    v_monthly.append(df[df['month']==i].sum()[1] / (df['month']==i).sum())

plt.figure(figsize=[9,2.5])
plt.plot(list(range(1,13,1)), v_monthly)
plt.xticks(ticks=list(range(1,13,1)))
plt.title("")
plt.xlabel("Month in a year")
plt.ylabel("Likelihood of Popularity")
plt.tight_layout()
plt.savefig("saved_plots/websearch_month.png")
plt.show()

# %%
# Plot google trends based on years
v_yearly = []
x = list(range(df['year'].min(),df['year'].max()+1,1))
for i in x:
    v_yearly.append(df[df['year']==i].sum()[1] / (df['year']==i).sum())

plt.figure(figsize=[9,2.5])
plt.plot(x, v_yearly)
plt.xticks(ticks=x)
plt.title("")
plt.xlabel("Year")
plt.ylabel("Google Trends Popularity")
plt.tight_layout()
plt.savefig("saved_plots/websearch_yearly.png")
plt.show()


# %%
# Yearly Data from NHANES
years = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
p = []
for y in years:
    df_google = step1_preprocess.read_sas(f"data/trends/DPQ_{y}.XPT")
    df_google = step1_preprocess.preprocess(df_google)
    proportion = df_google['Major depression'].sum() / df_google.shape[0]
    for i in range(24):
        p.append(proportion)
    print(y, ":", proportion)
    print("")


# %%
# Plot NHANES and Google data in one plot

fig, ax1 = plt.subplots(figsize=[9,2.5])
color = '#E0162B'
color_b = '#0052A5'
plt.vlines(x=list(range(0,len(m),12)), ymin=40, ymax=100, color="#DDDDDD", linestyles='--')
ax1.plot(df[0], df[1], label='Google Trends', c=color_b)
ax2 = ax1.twinx()
ax2.plot(df[0][:len(p)], p, label='NHANES', color=color)
ax1.set_ylabel("Google Trends Popularity", color=color_b)
ax1.tick_params(axis='y', labelcolor=color_b)
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim([0.01,0.06])
ax2.set_ylabel("NHANES Proportion", color=color)
plt.xticks(ticks=list(range(12,len(m),24)), rotation=45)
plt.tight_layout()
plt.savefig("saved_plots/websearch_timeseries.png")
plt.show()

# %%
