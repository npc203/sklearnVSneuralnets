import pandas as pd
from sklearn.preprocessing import LabelBinarizer,scale
from sklearn.metrics import classification_report as cr
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
import pickle
import numpy as np
from keras import backend as K
import time


train_df = pd.read_csv('train.csv')
#print(train_df.head())
train_X = train_df.drop(columns=['Severity'])
train_y = train_df[['Severity']]


le = LabelBinarizer()
le.fit(train_y)
pickle.dump(le,open("le.pickle","wb"))
train_X=scale(train_X)
train_y = le.transform(train_y)
#print(train_y)

ob1=KNeighborsClassifier(3)
ob2=SVC(kernel="rbf", C=0.025, probability=True)
ob3=DecisionTreeClassifier()
ob4=RandomForestClassifier()
ob5=AdaBoostClassifier()
ob6=GradientBoostingClassifier()
ob7=XGBClassifier()

algorithms = [('knc', ob1),('svc', ob2),('dtc', ob3),('rcf', ob4),('abc', ob5),('gbc', ob6),('xgbc', ob7)]
classifiers = [
KNeighborsClassifier(3),
SVC(kernel="rbf", C=0.025, probability=True),
DecisionTreeClassifier(),
RandomForestClassifier(),
AdaBoostClassifier(),
GradientBoostingClassifier(),
XGBClassifier(),
VotingClassifier(algorithms)
]

k_list = [2,3,4,5]
for classifier in classifiers:
    accu = []
    pipe = Pipeline(steps=[('classifier', classifier)])
    
    for j in k_list:
        accuracy = []
        kf = KFold(n_splits=j,shuffle=True)
        for (train, test), i in zip(kf.split(x, y), range(5)):
            pipe.fit(x[train], y[train])
            ypred=pipe.predict(x[test])
            accuracy.append(accuracy_score(y[test],ypred))
            print(classifier)
            print(cr(y[test],ypred))
        print("\nList of Accuracies:",accuracy,"\n")
        acc = sum(accuracy)/j
        print("\nAverage Accuracy:",acc,"\n")
        accu.append(acc)
    print("\nList of Average Accuracies for different k-folds:",accu,"\n")
    '''
    plt.plot(k_list,accu)
    plt.ylabel('Accuracy')
    plt.xlabel('K-Fold')
    plt.show()
    '''
    print("____________________________________________________________________________________________________________________________")

