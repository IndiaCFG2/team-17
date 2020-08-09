import nltk
import re
import pandas as pd
import numpy as np
import pickle
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk import pos_tag
from nltk.stem import PorterStemmer,WordNetLemmatizer
stop = stopwords.words('english')
from sklearn.preprocessing import LabelEncoder # convert target variable classes to numbers
from sklearn.naive_bayes import MultinomialNB


fp= open('C:/Users/akpsb/CFG/team-17/demo1/polls/amazon_cells_labelled.txt')
#fp = open("file")
c=9
a=[]
b=[]
for i, line in enumerate(fp):
    a.append(line.split('\t')[0])
    b.append(line.split('\t')[1][0])
    
fp.close()

def preprocess_text(x):
    clean_x=[]
    for i in x:
        st=str(i).lower()
        clean_sent=re.sub(r'\'','',st)
        clean_sent = re.sub(r'\W',' ',clean_sent)
        clean_sent = re.sub(r'\s+',' ',clean_sent)
        clean_x.append(clean_sent)
    return clean_x


df1=pd.DataFrame(a)
df2=pd.DataFrame(b)
cv = CountVectorizer()
df1_cv = cv.fit_transform(a).toarray()
cv_df = pd.DataFrame(df1_cv,columns=cv.get_feature_names())
cv_df["target"]=df2


le = LabelEncoder()

cv_df["target"]=le.fit_transform(cv_df["target"]) 

x_only_comments = cv_df.drop(columns='target') # Taking independent columns
y_category = cv_df['target']


model = MultinomialNB()

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x_only_comments,y_category,train_size=0.80)

model.fit(x_train,y_train)

model.predict(x_train)

y_train_pred=model.predict(x_train)


pickle.dump(model, open('model1.pkl','wb'))
model1 = pickle.load(open('model1.pkl','rb'))
