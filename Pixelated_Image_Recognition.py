import numpy as np
import pandas as pd 
import time
import seaborn as sb 
import matplotlib.pyplot as plt
from sklearn import tree,naive_bayes,neighbors
from sklearn.metrics import accuracy_score,confusion_matrix
train=pd.read_csv('mnist_test.txt')
features_train=np.array(train.drop(['label'],'columns'))
labels_train=np.array(train['label'])
clf=tree.DecisionTreeClassifier()
t1 = time.time()
clf=neighbors.KNeighborsClassifier()
clf.fit(features_train,labels_train)
t2 = time.time()
print("time", t2-t1)
t3 = time.time()
pre=clf.predict(features_train)
t4 = time.time()
print('time test', t4-t3)
print("prediction-", pre)
print("actual values-", labels_train)
acc=accuracy_score(pre,labels_train)
print("aacuracy=",acc)

x=147
print("predicted text", pre[x])
print("actual digit", labels_train[x])
digit=features_train[x]
digit_pixels=digit.reshape(28,28)
plt.imshow(digit_pixels, cmap='gray_r')
plt.show()

cm=confusion_matrix(labels_train,pre)
print("confusion matrix")
print(cm)

axis=plt.subplot()
sb.heatmap(cm,ax=axis,annot=True)
axis.set_xlabel("prediction")
axis.set_ylabel("actual numbers")
axis.set_title("Kneighors")
plt.show()