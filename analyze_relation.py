import numpy as np
import pandas as pd
import scipy.stats as ss
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

sns.set_context("talk", font_scale=0.8)  # 修正 font_scale 参数的使用
df = pd.read_csv("bank.csv", sep=';')
df['loan']=df['loan'].map({'yes':1,'no':0})
df['housing'] = df['housing'].map({'yes': 1, 'no': 0})
df['y'] = df['y'].map({'yes': 1, 'no': 0})
df['month']=LabelEncoder().fit_transform(df['month'])
df['poutcome']=LabelEncoder().fit_transform(df['poutcome'])
df['default'] = df['default'].map({'yes': 1, 'no': 0})
newjob = LabelEncoder().fit_transform(df['job'])
df['job'] = newjob

# 4.2 marrage字段处理，经过前面的数据分析，可以把单身和离婚归为一类
df['marital'] = df['marital'].map({'married': 1, 'single': 0, 'divorced': 0})
# 4.3 education字段处理，数值化
newedu = LabelEncoder().fit_transform(df['education'])
df['education'] = newedu
# 选择特定的特征
features = ['default','balance','loan','housing','y']
corr_matrix = df[features].corr()

# 绘制热图
sns.heatmap(corr_matrix, vmin=-1, vmax=1, cmap=sns.color_palette("RdBu", n_colors=128), annot=True)
plt.show()
