import numpy as np
import pandas as pd 
from sklearn import tree,naive_bayes
from sklearn.metrics import accuracy_score
train=pd.read_csv('mnist_test.txt')
features_train=np.array(train.drop(['label'],'columns'))
labels_train=np.array(train['label'])
clf=tree.DecisionTreeClassifier()
clf.fit(features_train,labels_train)
pre=clf.predict(features_train)
print("prediction-", pre)
print("actual values-", labels_train)
acc=accuracy_score(pre,labels_train)
print("aacuracy=",acc)