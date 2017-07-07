 

from self import preprocess
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.grid_search import GridSearchCV

features_train,features_test,labels_train,labels_test,mail=preprocess()
grid={'C':[0.1,1,10,100],'gamma':[0.001,0.01,0.1,1,10]}
clf=GridSearchCV(SVC(kernel="rbf"),grid)
clf.fit(features_train,labels_train)
x=clf.predict(features_test)
print clf.predict(mail)
print clf.get_params()
print f1_score(labels_test,x,average="micro")
print confusion_matrix(labels_test,x)

