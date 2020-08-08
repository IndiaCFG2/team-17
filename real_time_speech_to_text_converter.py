import speech_recognition as sr
from googletrans import Translator

temp=sr.Recognizer()
with sr.Microphone() as source:
    print("Say Something!")
    audio=temp.listen(source)
    print('Done')
    
text=temp.recognize_google(audio)
speech_trans=Translator()
translated_output=speech_trans.translate(text,dest='en')

print(translated_output.text)
print("Text Printed!")
