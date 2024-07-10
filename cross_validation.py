import numpy as np
import pandas as pd
import scipy.stats as ss
import matplotlib.pyplot as plt
import seaborn as sns


df=pd.read_csv("bank.csv",sep=';')
#数据转换
df['y'] = df['y'].map({'yes': 1, 'no': 0})

mar_indeces=df.groupby("marital").indices
single_value=df["y"].iloc[mar_indeces["single"]].values
divorced_value=df["y"].iloc[mar_indeces["divorced"]].values
print(ss.ttest_ind(single_value,divorced_value)[1])

mar_keys=list(mar_indeces.keys())
mar_t_mat=np.zeros([len(mar_keys),len(mar_keys)])
for i, itemi in enumerate(mar_keys):
    for j, itemj in enumerate(mar_keys):
        p_value=ss.ttest_ind(df["y"].iloc[mar_indeces[itemi]].values, df['y'].iloc[mar_indeces[itemj]].values)[1]
        if p_value<0.05:
            mar_t_mat[i,j]=-1
        else:
            mar_t_mat[i,j]=p_value
sns.heatmap(mar_t_mat,xticklabels=mar_keys,yticklabels=mar_keys)
plt.show()