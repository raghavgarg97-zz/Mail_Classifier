#!/usr/bin/python

from nltk.stem.snowball import SnowballStemmer
import string

def parseOutText(f):
    f.seek(0)  
    all_text = f.read()
    content=all_text.split("X-FileName:")
    words = ""
    stemmer=SnowballStemmer("english")
    text_string = content[1].translate(string.maketrans("", ""), string.punctuation)
    s=text_string.split()
    a= content[0].split("X-From:")
    b=a[1].split("X-To:")
    c=b[0].translate(string.maketrans("", ""), string.punctuation)
    d=b[1].split("X-cc:")
    e=d[0].translate(string.maketrans("", ""), string.punctuation)
    g=s+c.split()+e.split()
    return ' '.join(g) 
