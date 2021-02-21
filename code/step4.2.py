#!/usr/bin/env python
# coding: utf-8

# In[33]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr


# In[73]:


data = pd.read_csv('data_for_neha.csv')
data.head()


# In[74]:


corr = data.corr()
plt.figure(figsize=(12,8))
sns.heatmap(corr)


# In[75]:


man = pd.read_csv('manufacture.csv')
man = man.replace(':', '')
man['Value'] = pd.to_numeric(man['Value'])
man.head()


# In[76]:


data['Manufacturing before COVID'] = data['location'].apply(lambda x: man[(man['GEO'] == x) 
                                                                          & (man['NACE_R2'] == 'Manufacturing') 
                                                                          & (man['TIME'] <= '2019')]['Value'].sum()
                                                            / len(man[(man['GEO'] == x) 
                                                                          & (man['NACE_R2'] == 'Manufacturing') 
                                                                          & (man['TIME'] <= '2019')]['Value']))
data['Manufacturing during COVID'] = data['location'].apply(lambda x: man[(man['GEO'] == x) 
                                                                          & (man['NACE_R2'] == 'Manufacturing') 
                                                                          & (man['TIME'] >= '2020')]['Value'].sum()
                                                            / len(man[(man['GEO'] == x) 
                                                                          & (man['NACE_R2'] == 'Manufacturing') 
                                                                          & (man['TIME'] >= '2020')]['Value']))
data['Manufacturing'] = data['location'].apply(lambda x: man[(man['GEO'] == x) 
                                                                          & (man['NACE_R2'] == 'Manufacturing')]['Value'].sum()
                                                            / len(man[(man['GEO'] == x) 
                                                                          & (man['NACE_R2'] == 'Manufacturing')]['Value']))

data['Manufacturing Change'] = data['Manufacturing during COVID'] - data['Manufacturing before COVID']

data = data[~(data['Manufacturing Change'] == 0)]

data


# In[77]:


fig, ax = plt.subplots()
df = data
x = 'distance'
y = 'Manufacturing before COVID'
df.plot(x, y, kind='scatter', ax=ax, figsize=(10,6))
df_by_state = df.set_index('location')
for k, v in df_by_state[[x, y]].iterrows():
    ax.annotate(k, v)

z = np.polyfit(df[x], df[y], 1)
p = np.poly1d(z)
corr, _ = pearsonr(df[x],df[y])
print(corr)
plt.plot(df[x],p(df[x]),"r-", 
         label='y=' + str(round(z[0], 7)) + 'x + ' + str(round(z[1], 2)) + ', R=' + str(round(corr, 3)))
plt.xlabel('Time-Warped Distance')
# plt.ylabel('Difference in Calculated and Reported Case Count')
plt.title('Manufacturing before COVID vs. Time-Warped Distance')
plt.grid(True)
plt.legend()


# In[85]:


from sklearn.neighbors import LocalOutlierFactor
lof = LocalOutlierFactor(4)
yhat = lof.fit_predict(df[[x, y]])


# In[87]:


df['Distance from LinReg'] = df.apply(lambda a: abs(z[0]*a[x] + z[1] - a[y]), axis=1)
df

