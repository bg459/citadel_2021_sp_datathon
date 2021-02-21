#!/usr/bin/env python
# coding: utf-8

# In[202]:


import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


# In[227]:


cases = pd.read_csv('1_owid/owid-covid-data.csv')
deaths = pd.read_csv('country_ages.csv')
deaths['Value'] = deaths['Value'].apply(lambda x: x.replace(',', ''))
deaths['Value'] = deaths['Value'].apply(lambda x: x.replace(':', ''))
late_dates = ['2021W01', '2021W02', '2021W03', '2021W04', '2021W05']
deaths = deaths[~deaths['TIME'].isin(late_dates)]

inval_groups = ['Total']
deaths_total = deaths[deaths['AGE'].isin(inval_groups)]
deaths = deaths[~deaths['AGE'].isin(inval_groups)]
deaths_total['Value'] = pd.to_numeric(deaths_total['Value'])
deaths['Value'] = pd.to_numeric(deaths['Value'])
deaths.head()


# In[228]:


cases.head()


# In[229]:


states = deaths.GEO.unique()

df = pd.DataFrame({'State': states})
df['Total Deaths'] = df['State'].apply(lambda x: cases[(cases['location'] == x) & (cases['date'] == '2021-01-04')]['total_deaths'].sum())
group_mappings = {
    'Deaths less than 25 years': ['Less than 5 years', 'From 5 to 9 years', 'From 10 to 14 years', 'From 15 to 19 years', 'From 20 to 24 years'],
    'Deaths from 25 to 44 years': ['From 25 to 29 years', 'From 30 to 34 years', 'From 35 to 39 years', 'From 40 to 44 years'],
    'Deaths from 45 to 64 years': ['From 45 to 49 years', 'From 50 to 54 years', 'From 55 to 59 years', 'From 60 to 64 years'],
    'Deaths from 65 to 74 years': ['From 65 to 69 years', 'From 70 to 74 years'],
    'Deaths 75 years and over': ['From 75 to 79 years', 'From 80 to 84 years', 'From 85 to 89 years', '90 years or over']
}
for g in group_mappings:
    df[g] = df['State'].apply(lambda x: deaths[(deaths['GEO'] == x) 
                                               & (deaths['AGE'].isin(group_mappings[g]))]['Value'].sum()
                             * df[df['State'] == x]['Total Deaths'].sum() 
                              / deaths_total[deaths_total['GEO'] == x]['Value'].sum())
    
df = df[df['State'].isin(cases.location.unique().tolist())]
# df


# In[230]:


ifr = {
    25: 0.0000972,
    44: 0.00116,
    64: 0.00939,
    74: 0.0487,
    100: 0.142
}
# df['Total Deaths'] = df.apply(lambda x: x['Deaths less than 25 years']
#                                    + x['Deaths from 25 to 44 years']
#                                    + x['Deaths from 45 to 64 years']
#                                    + x['Deaths from 65 to 74 years']
#                                    + x['Deaths 75 years and over'], axis=1)
# df['Reported Deaths'] = df['State'].apply(lambda x: cases[cases['location'] == x].iloc[0]['Total Deaths'])

df['Calculated Cases'] = df.apply(lambda x: x['Deaths less than 25 years']/ifr[25]
                                   + x['Deaths from 25 to 44 years']/ifr[44]
                                   + x['Deaths from 45 to 64 years']/ifr[64]
                                   + x['Deaths from 65 to 74 years']/ifr[74]
                                   + x['Deaths 75 years and over']/ifr[100]
                                  , axis=1)


# In[231]:


df['Reported Cases'] = df['State'].apply(lambda x: cases[(cases['location'] == x) & (cases['date'] == '2021-01-04')]['total_cases'].sum())
df['Cases Difference'] = abs(df['Calculated Cases'] - df['Reported Cases'])/df['Reported Cases']
# adjust for countries with too small sample size
df = df[df['Total Deaths'] > 300]
df = df[~(df['State'] == 'Luxembourg')]
# df


# In[235]:


tests = pd.read_csv('2_ecdc/testing.csv')
df['Tests Performed'] = df['State'].apply(lambda x: tests[tests['country'] == x]['testing_rate'].sum()
                                          /len(tests[tests['country'] == x]['testing_rate']))
corr, _ = pearsonr(df['Tests Performed'], df['Cases Difference'])
print(corr)


# In[238]:


fig, ax = plt.subplots()
df.plot('Tests Performed', 'Cases Difference', kind='scatter', ax=ax, figsize=(10,6))
df_by_state = df.set_index('State')
for k, v in df_by_state[['Tests Performed', 'Cases Difference']].iterrows():
    ax.annotate(k, v)

z = np.polyfit(df['Tests Performed'], df['Cases Difference'], 1)
p = np.poly1d(z)
plt.plot(df['Tests Performed'],p(df['Tests Performed']),"r-", 
         label='y=' + str(round(z[0], 7)) + 'x + ' + str(round(z[1], 2)) + ', R=' + str(round(corr, 3)))
plt.xlabel('Average Weekly Tests Performed per 100k')
plt.ylabel('Difference in Calculated and Reported Case Count')
plt.title('Difference in Cases Metrics vs. Testing Volume (EU)')
plt.grid(True)
plt.legend()


# In[239]:


df

