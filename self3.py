from self import preprocess
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.grid_search import GridSearchCV

features_train,features_test,labels_train,labels_test,mail=preprocess()
clf=RandomForestClassifier(min_samples_split=5)
clf.fit(features_train,labels_train)
x=clf.predict(features_test)
print clf.predict(mail)
print f1_score(labels_test,x,average="micro")
print confusion_matrix(labels_test,x)

