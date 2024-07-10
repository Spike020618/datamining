from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.model_selection import train_test_split
#import os
#os.environ['PATH'] += os.pathsep+"D:/Graphviz/bin/"

from preprocessing import preprocessing

def modeling(features,label):
    #切分训练集测试集验证集
    f_v=features.values
    l_v=label.values
    X_tt,X_validation,Y_tt,Y_validation=train_test_split(f_v,l_v,test_size=0.2)
    X_train,X_test,Y_train,Y_test=train_test_split(X_tt,Y_tt,test_size=0.2)
    print(len(X_train),len(X_test),len(X_validation))

    models=[]
    #models.append(("KNN",KNeighborsClassifier(n_neighbors=3)))
    #models.append(("GaussianNB",GaussianNB()))
    #models.append(("BernoulliNB",BernoulliNB()))
    models.append(("DecisionTreeClassifier",DecisionTreeClassifier()))
    models.append(("DecisionTreeEntropy",DecisionTreeClassifier(criterion='entropy')))
    for clf_name, clf in models:
        clf.fit(X_train,Y_train)
        datasetcat = ['train', 'valid', 'test']
        xy_lst=[(X_train,Y_train),(X_validation,Y_validation),(X_test,Y_test)]
        print(f'\n[{clf_name}]\n\tACC\t\tREC\t\tF1')
        for i, item in enumerate(xy_lst):
            X_part, Y_part=item[0], item[1]
            y_pred=clf.predict(X_part)
            print(f'{datasetcat[i]}\t{accuracy_score(Y_part, y_pred):.10f}\t{recall_score(Y_part, y_pred):.10f}\t{f1_score(Y_part, y_pred):.10f}')
            #f_names=features.columns.values
            #dot_data=export_graphviz(clf,out_file=None,feature_names=f_names,class_names=["YES","NO"],filled=True,rounded=True,special_characters=True)
            #graph=pydotplus.graph_from_dot_data(dot_data)
            #graph.write_pdf('dt_tree2.pdf')


def main():
    features,label=preprocessing(aging=True,bal=True,dur=True,pdays=True)
    print(f'Feature: {features.columns.tolist()}')
    modeling(features,label)


main()