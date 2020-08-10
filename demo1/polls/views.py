from django.template import loader
from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import get_object_or_404, render
from django.urls import reverse
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from scipy.io.wavfile import read,write
from googletrans import Translator
import pickle
from django.core.files.storage import FileSystemStorage
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import re
from rake_nltk import Rake
from pydub import AudioSegment
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

import os

cv=CountVectorizer()

from .models import Question, Choice
model1 = pickle.load(open('C:/Users/akpsb/CFG/team-17/demo1/polls/modelu.pkl', 'rb'))

def index(request):
    latest_question_list = Question.objects.order_by('-pub_date')[:5]
    context = {'latest_question_list': latest_question_list}
    return render(request, 'polls/index.html', context)


def detail(request, question_id):
  try:
    question = Question.objects.get(pk=question_id)
  except Question.DoesNotExist:
    raise Http404("Question does not exist")
  return render(request, 'polls/detail.html', { 'question': question })

def results(request, question_id):
  question = get_object_or_404(Question, pk=question_id)
  return render(request, 'polls/results.html', { 'question': question  })

def vote(request, question_id):
    
    question = get_object_or_404(Question, pk=question_id)
    try:
        selected_choice = question.choice_set.get(pk=request.POST['choice'])
    except (KeyError, Choice.DoesNotExist):
        
        return render(request, 'polls/detail.html', {
            'question': question,
            'error_message': "You didn't select a choice.",
        })
    else:
        selected_choice.votes += 1
        selected_choice.save()
        return HttpResponseRedirect(reverse('polls:results', args=(question.id,)))

def resultsData(request, obj):
    votedata = []

    question = Question.objects.get(id=obj)
    votes = question.choice_set.all()

    for i in votes:
        votedata.append({i.choice_text:i.votes})

    print(votedata)
    return JsonResponse(votedata, safe=False)

@csrf_exempt
def ace( request ):
    temp=(request.POST.get('input'))
    print(request.POST.get('input'))
    speech_trans=Translator()



    


    translated_output=speech_trans.translate(temp,dest='en')
    print(translated_output)
    #res=classify(translated_output.text)
    #print(res)
    sentiment = modal(translated_output.text)
    obj={'output':translated_output.text,'sentiment':sentiment}

    
    return JsonResponse(obj,safe=False)


def preprocess_text(x):
    clean_x=[]
    for i in x:
        st=str(i).lower()
        clean_sent=re.sub(r'\'','',st)
        clean_sent = re.sub(r'\W',' ',clean_sent)
        clean_sent = re.sub(r'\s+',' ',clean_sent)
        clean_x.append(clean_sent)
    return clean_x

def classify(test_input):
    test_item=pd.DataFrame([test_input])
    test_item=test_item[0]
    test_item=preprocess_text(test_item)
    print(test_item)
    print(cv.vocabulary)
    new_cv=cv.transform(test_item).toarray()
    print(new_cv)
    if model1.predict(new_cv)[0]==0:
        return "-ve"
    return "+ve"




@csrf_exempt
def data(request):
    df=pd.read_csv('C:/Users/akpsb/CFG/team-17/demo1/polls/feedback.txt')
    a=[]
    n=0
    p=0
    b=""
    for i, line in enumerate(fp):
        a.append(line.split('\t')[0])
        b+=" "+line.split('\t')[0]
    for i in a:
        c=classify(i)
        if(c=="+ve"):
            p+=1
        else:
            n+=1
    sent={'yes':p,"no":n}
    r = Rake() # Uses stopwords for english from NLTK, and all puntuation characters.

    r.extract_keywords_from_text(b)

    ranked=r.get_ranked_phrases()

    print(ranked) # To get keyword phrases ranked highest to lowest.

    return JsonResponse(sent,safe=False)


    



@csrf_exempt
def speech(request):
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        print(uploaded_file_url)
        
        s=os.getcwd()
        sound =AudioSegment.from_file(uploaded_file_url)
        sound.export("temporary_output.wav",format="wav")
        audio="temporary_output.wav"
        r=sr.Recognizer()
        with sr.AudioFile(audio) as source:
            audio1=r.record(source)
            text=r.recognize_google(audio1)
            speech_trans=Translator()
            translated_output=speech_trans.translate(text,dest='en')
            print(translated_output.text)
            output=translated_output.text
            return JsonResponse(output,safe=False)


def modal(input):
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

    from  sklearn.feature_extraction.text import CountVectorizer

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
    

    test_input=input
    return(classify(test_input))
    """
    test_input=input("Enter a new msg text : ")
    print(classify(test_input))
 """
  

