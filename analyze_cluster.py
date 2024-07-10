import pandas as pd
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier


sns.set_context(font_scale=1.5)

df=pd.read_csv("bank.csv",sep=';')

age=df['age']
sns.barplot(x=list(range(len(age))),y=age.sort_values())
#按照拐点选取分界点
plt.show()

#使用分类器计算最佳分界点（基尼系数）

clf=DecisionTreeClassifier(criterion='gini',random_state=2)#第二个参数设置分裂点数

clf.fit(np.array(age).reshape(-1,1),df['y'])
threshod=clf.tree_.threshold[0]
print(f"最佳分界点：{threshod}")