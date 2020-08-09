from django.template import loader
from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import get_object_or_404, render
from django.urls import reverse
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from googletrans import Translator
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import re


from .models import Question, Choice
model1 = pickle.load(open('C:/Users/akpsb/CFG/team-17/demo1/polls/model1.pkl', 'rb'))

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
    temp=(request.POST.get('answer'))
    speech_trans=Translator()
    translated_output=speech_trans.translate(temp,dest='en')
    print(translated_output)
    res=classify(translated_output.text)
    print(res)
    

    return JsonResponse(translated_output.text,safe=False)


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
    cv=CountVectorizer()
    new_cv=cv.transform(test_item).toarray()
    if model1.predict(new_cv)[0]==0:
        return "-ve"
    return "+ve"