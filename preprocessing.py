import pandas as pd
from sklearn.utils import resample
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder

def preprocessing(aging=False,bal=False,dur=False,camp=False,pdays=False):
    '''
    :param aging:
    :param bal:
    :param dur:
    features:age(scaler),job(num),marital(0,1),education(num),default(0,1),loan(0,1),balance(scaler),
    contact(0,1),campaign,pdays
    :return:
    '''
    df=pd.read_csv('./bank.csv', sep=';')
    # 将标签为 "yes" 和 "no" 分开
    yes_class = df[df['y'] == "yes"]
    no_class = df[df['y'] == 'no']

    # 对多数类样本进行下采样，假设选择与少数类样本相同数量的多数类样本
    # 这里假设少数类 "yes" 有 100 个样本，我们从多数类 "no" 中随机抽取 100 个样本
    # 可以根据实际情况调整抽样的数量
    no_class_downsampled = resample(no_class, replace=False,  # 不放回抽样
                                    n_samples=len(yes_class),  # 与少数类样本相同数量
                                    random_state=42)  # 设置随机种子保证可复现性
    # 合并下采样后的多数类样本和少数类样本
    df = pd.concat([yes_class, no_class_downsampled])

    # 1.得到标注
    label=df["y"].map({"yes":1,"no":0})
    df=df.drop("y",axis=1)#axis如果不等于1会以行进行删除

    # 2.清洗数据（异常值，空值）
    df=df.dropna(axis=0,how='any')#0表示行，all表示全部为空值才删
    #df=df[df["xxx"]>=a&df["yyy"]<=b]

    # 3.特征选择

    # 4.特征处理
    # 建立标准化列表，批量处理标准化需求
    scaler_lst=[aging,bal,dur,camp,pdays]
    colum_lst=["age",'balance','duration','campaign','pdays']
    for i, item in enumerate(scaler_lst):
        if not item:
            df[colum_lst[i]]=\
            MinMaxScaler().fit_transform(df[colum_lst[i]].values.reshape(-1,1)).reshape(1,-1)[0]
        else:
            df[colum_lst[i]]= \
            StandardScaler().fit_transform(df[colum_lst[i]].values.reshape(-1, 1)).reshape(1, -1)[0]
    # 4.1 job字段数值化
    newjob=LabelEncoder().fit_transform(df['job'])
    df['job']=newjob
    newpoutcome = LabelEncoder().fit_transform(df['poutcome'])
    df['poutcome'] = newpoutcome
    newpday = LabelEncoder().fit_transform(df['month'])
    df['month'] = newpday
    # 4.2 marrage字段处理，经过前面的数据分析，可以把单身和离婚归为一类
    df['marital'] = df['marital'].map({'married': 1, 'single': 0, 'divorced': 0})
    #4.3 education字段处理，数值化
    newedu=LabelEncoder().fit_transform(df['education'])
    df['education']=newedu
    # 4.4 违约处理
    df['default']=df['default'].map({'yes':1,'no':0})
    # 4.5 贷款处理,房贷处理，经过分析，认为贷款和房贷不可合并,但是房贷和y的相关性太小，故删去housing
    df['loan']=df['loan'].map({'yes':1,'no':0})
    df['housing'] = df['housing'].map({'yes': 1, 'no': 0})
    # 4.6 联系方式，经过分析,可以把移动电话和座机记为1，把未知记为0
    df['contact']=df['contact'].map({'unknown':0,'telephone':1,'cellular':1})
    #df = df.drop(columns=['month', 'day', 'duration', 'previous', 'poutcome', 'housing'])
    # 4.8 经过相关分析，我认为，pdays可以分为联系过和没联系过
    df['pdays'] = df['pdays'].apply(lambda x: 1 if x == -1 else 0)
    # 4.7 经过相关分析，我认为，day和month,联系次数可以去掉
    df = df.drop(columns=['age','job','education','default','balance','marital','campaign','pdays'])

    #4.9 经过相关分析，我认为，poutcomes和previous,pdays相关性很高，因此可以去掉previous，poutcome,保留pdays
    return df,label