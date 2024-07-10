from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB,BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
#import os
#os.environ['PATH'] += os.pathsep+"D:/Graphviz/bin/"

from preprocessing import preprocessing


def modeling(features,label):

    f_v=features.values
    l_v=label.values

    models=[]
    models.append(("KNN",KNeighborsClassifier(n_neighbors=3)))
    models.append(("GaussianNB",GaussianNB()))
    models.append(("BernoulliNB",BernoulliNB()))
    # model advanece solution
    models.append(("DecisionTreeClassifier",DecisionTreeClassifier(criterion='gini',max_features='sqrt',max_depth=None,min_samples_leaf=1,min_samples_split=5,class_weight='balanced')))
    models.append(("RandomForestClassifier", RandomForestClassifier(n_estimators=100,criterion='gini',max_features='sqrt',max_depth=None,min_samples_leaf=1,min_samples_split=5,class_weight='balanced')))

    for clf_name,clf in models:
        #xy_lst=[(X_train,Y_train),(X_test,Y_test)]
        scoring = ['accuracy', 'matthews_corrcoef']
        results = cross_validate(clf, f_v, l_v, cv=5, scoring=scoring)
        print(f'\n[{clf_name}]')
        print("ACC\t\t",results['test_accuracy'])
        print("MCC\t\t", results['test_matthews_corrcoef'])
        print("ACC means\t",results['test_accuracy'].mean())
        print("MCC means\t",results['test_matthews_corrcoef'].mean())
        #f_names=features.columns.values
        #dot_data=export_graphviz(clf,out_file=None,feature_names=f_names,class_names=["YES","NO"],filled=True,rounded=True,special_characters=True)
        #graph=pydotplus.graph_from_dot_data(dot_data)
        #graph.write_pdf('dt_tree2.pdf')

def main():
    features,label = preprocessing(aging=True,bal=True,dur=True,pdays=True)
    print(f'Feature: {features.columns.tolist()}')
    modeling(features,label)


main()
