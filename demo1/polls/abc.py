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
print(b)
a.append('This could have been better.')
b.append('0')

df1=pd.DataFrame(a)
df2=pd.DataFrame(b)

df1.head()

df2.head()

a

def preprocess_text(x):
    clean_x=[]
    for i in x:
        st=str(i).lower()
        clean_sent=re.sub(r'\'','',st)
        clean_sent = re.sub(r'\W',' ',clean_sent)
        clean_sent = re.sub(r'\s+',' ',clean_sent)
        clean_x.append(clean_sent)
    return clean_x

new_df1=preprocess_text(df1)

new_df1=pd.DataFrame(new_df1)

new_df1.head()

df1.tail()

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()

type(a)

df1_cv = cv.fit_transform(a).toarray()

cv_df = pd.DataFrame(df1_cv,columns=cv.get_feature_names())

cv_df.head()

cv_df["target"]=df2

from sklearn.preprocessing import LabelEncoder # convert target variable classes to numbers

le = LabelEncoder()

cv_df["target"]=le.fit_transform(cv_df["target"]) 

cv_df.head(3)

x_only_comments = cv_df.drop(columns='target') # Taking independent columns
y_category = cv_df['target']

from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x_only_comments,y_category,train_size=0.80)

model.fit(x_train,y_train)

model.predict(x_train)

y_train_pred=model.predict(x_train)

from sklearn.metrics import precision_score,recall_score,accuracy_score,auc,roc_curve

precision_score(y_train,y_train_pred)

x_train



def classify(test_input):
    test_item=pd.DataFrame([test_input])
    test_item=test_item[0]
    test_item=preprocess_text(test_item)
    print(test_item)
    new_cv=cv.transform(test_item).toarray()
    print(new_cv)
    print(cv.vocabulary)
    if model.predict(new_cv)[0]==0:
        return "-ve"
    return "+ve"
    

test_input=input("Enter a new msg text : ")
print(classify(test_input))

test_input=input("Enter a new msg text : ")
print(classify(test_input))

model.predict(x_test)

y_test_pred = model.predict(x_test)
precision_score(y_test,y_test_pred)

from matplotlib import pyplot as plt
 
plt.bar([1],[40],label="No",width=1)
plt.bar([2],[60],label="Neutral", color='r',width=1)
plt.bar([3],[40],label="Yes",width=1)
plt.legend()
plt.xlabel('Days')
plt.ylabel('Distance (kms)')
plt.title('Information')
plt.show()

from matplotlib import pyplot as plt
import numpy as np
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.axis('equal')
langs = ['Positive','negative']
students = [60,40]
ax.pie(students, labels = langs,autopct='%1.2f%%')
plt.show()

x_test


import pickle

pickle.dump(model, open('modelu.pkl','wb'))
model = pickle.load(open('modelu.pkl','rb'))

