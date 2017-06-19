
#!/usr/bin/python

import numpy
import os
from sklearn import cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif
from parse_out_email_text import parseOutText
from nltk.corpus import stopwords
def preprocess():
    """ 
        this function takes a pre-made list of email texts (by default word_data.pkl)
        and the corresponding authors (by default email_authors.pkl) and performs
        a number of preprocessing steps:
            -- splits into training/testing sets (10% testing)
            -- vectorizes into tfidf matrix
            -- selects/keeps most helpful features

        after this, the feaures and labels are put into numpy arrays, which play nice with sklearn functions

        4 objects are returned:
            -- training/testing features
            -- training/testing labels

  """
    wordlist=[]
    
    f1=open("../text_learning/newdatabase.txt","r")
  
    for path in f1:
        path = os.path.join('..', path[:-1])
	print path
        email=open(path,"r")
        txt=parseOutText(email)
        wordlist.append(txt)
    a=[1]*(30)
    b=[2]*26
    c=[3]*(13)
    e=a+b+c
    sw=stopwords.words("english")
    features_train,features_test, labels_train, labels_test = cross_validation.train_test_split(wordlist,e, test_size=0.3, random_state=42)
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                 stop_words=sw)
    features_train_transformed = vectorizer.fit_transform(features_train)
    features_test_transformed  = vectorizer.transform(features_test)
    print"length:",len( vectorizer.get_feature_names())


    ### feature selection, because text is super high dimensional and 
    ### can be really computationally chewy as a result
    selector = SelectPercentile(f_classif, percentile=10)
    selector.fit(features_train_transformed, labels_train)
    features_train_transformed = selector.transform(features_train_transformed).toarray()
    features_test_transformed= selector.transform(features_test_transformed).toarray()
  ### info on the data
    print "no. type 1  emails:", 30
    print "no. of type 2 emails:", 26
    print "no. of type3 emails",13
    
    return features_train_transformed, features_test_transformed, labels_train, labels_test


preprocess()
