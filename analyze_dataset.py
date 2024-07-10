import numpy as np
import pandas as pd

df=pd.read_csv('bank.csv',sep=';')

#age分析,去掉空值，未成年去掉发现没有未成年
def age_analysis():
    age=df["age"].dropna()
    print(age.mean(),age.std(),age.max(),age.min(),age.median(),age.skew(),age.kurt())
    print(age.shape)

    #查看年龄段分布
    print(np.histogram(age.values,bins=np.arange(19,90,10)))

#分析工作，需要数值化
def jobs_analysis():
    job=df["job"].dropna()
    jobs_count=df["job"].value_counts()
    print(jobs_count)


#分析婚姻状态，共三种，可结合课上所学，分为已婚和未婚
def marriage_analysis():
    marriage=df["marital"].dropna()
    marriage_count=df["marital"].value_counts()
    print(marriage_count)
    #清异常值可以找到后再，ss=ss.where(job=!"xxx").dropna()

def balance_analysis():
    balance=df["balance"].dropna()
    print(df['balance'].value_counts())
    print(balance.mean(),balance.max(),balance.min())

def loan_analysis():
    df['loan'] = df['loan'].map({'yes': 1, 'no': 0})
    df['housing'] = df['housing'].map({'yes': 1, 'no': 0})
    same_values = (df['loan'] == df['housing'])
    df_filtered = df[~same_values]
    print(len(df_filtered))
    index=df_filtered.index
    print(df.loc[index, 'loan'].value_counts())

def duration_analysis():
    duration=df['duration'].dropna()
    print(df['duration'].value_counts(),duration.mean(),duration.max(),duration.min())

def lastcontact_analysis():
    lastcontact=df['pdays'].dropna()
    print(df['pdays'].value_counts(),lastcontact.mean(),lastcontact.max(),lastcontact.min())

age_analysis()
jobs_analysis()
marriage_analysis()
balance_analysis()
loan_analysis()
duration_analysis()
lastcontact_analysis()