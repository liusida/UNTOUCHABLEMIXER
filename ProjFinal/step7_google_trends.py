
#%%
import matplotlib.pyplot as plt
import pandas as pd
#%%
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

plt.plot(v)

# %%
df = pd.DataFrame([m,v]).T
df['month'] = df[0].str[-2:].astype(int)
# %%
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
plt.figure(figsize=[9,2.5])
plt.vlines(x=list(range(0,len(m),12)), ymin=40, ymax=100, color="#DDDDDD", linestyles='--')
plt.plot(df[0], df[1])
plt.xticks(ticks=list(range(0,len(m),12)), rotation=45)
plt.ylabel("Popularity")
plt.tight_layout()
plt.savefig("saved_plots/websearch_timeseries.png")
plt.show()
# %%
