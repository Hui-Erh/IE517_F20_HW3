# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 10:19:58 2021

@author: lenovo
"""

import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
sns.set(color_codes=True)

# import datasets
df = pd.read_csv('HY_Universe_corporate bond.csv', sep = ',')

# To determine how many of the columns of data are numeric or categorical.
df.info()
df.describe()
print(df.isnull().sum())

# Label Encode object data
obj_column = df.dtypes[df.dtypes == 'object'].index #列出所有'object'的index欄位
mapingdf = pd.DataFrame()
df_baseline = df.copy()
for column in obj_column:
    labelencoder = LabelEncoder()
    df_baseline[column] = labelencoder.fit_transform(df[column])
    mapingdf[column] = df_baseline[column]
    mapingdf['_'+column] =  labelencoder.inverse_transform(df_baseline[column])

labels = df_baseline['weekly_median_ntrades']
df_baseline = df_baseline.drop(['weekly_median_ntrades'], axis=1)

# Standardization
from sklearn.preprocessing import StandardScaler
int_column = df.dtypes[df.dtypes == 'int'].index
for column in int_column:
    scaler = StandardScaler()
    df_baseline[column] = scaler.fit_transform(df[column].values.reshape(-1, 1))

from sklearn.preprocessing import StandardScaler
float_column = df.dtypes[df.dtypes == 'float'].index
for column in float_column:
    scaler = StandardScaler()
    df_baseline[column] = scaler.fit_transform(df[column].values.reshape(-1, 1))

int_column = df.dtypes[df.dtypes == 'int64'].index | df.dtypes[df.dtypes == 'float64'].index

for column in int_column:    
    plt.figure(figsize=(16,4))

    plt.subplot(1,3,1)
    sns.distplot(df[column], kde_kws={'bw':0.1})
    plt.xlabel(column)
    plt.ylabel('Density')
    plt.title(f'{column}  Distribution')

    plt.subplot(1,3,2)
    sns.boxplot(x='weekly_median_ntrades', y=column, data =df, showmeans=True)
    plt.xlabel('Target')
    plt.ylabel(column)
    plt.title(f'{column}  Distribution')

    plt.subplot(1,3,3)
    counts, bins = np.histogram(df[column], bins=20, normed=True)
    cdf = np.cumsum (counts)
    plt.plot (bins[1:], cdf/cdf[-1])
    #plt.xticks(range(15,100,5))
    plt.yticks(np.arange(0,1.1,.1))
    plt.title(f'{column}  cdf')
    plt.show()
    print()

# Quantiles
for column in int_column:
    print(f'For {column}:')

    print('Min:', df[column].quantile(q = 0))
    print('1º Quartile:', df[column].quantile(q = 0.25))
    print('2º Quartile:', df[column].quantile(q = 0.50))
    print('3º Quartile:', df[column].quantile(q = 0.75))
    print('Max:', df[column].quantile(q = 1.00),'\n')
    
    
print("My name is Hui-Erh Chai_Angela")
print("My NetID is: 674939884")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")

