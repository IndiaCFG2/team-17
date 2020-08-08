import speech_recognition as sr

temp=sr.Recognizer()
with sr.Microphone() as source:
    print("Say Something!")
    audio=temp.listen(source)
    print('Done')
    
text=temp.recognize_google(audio)

print(text)
print("Text Printed!")
print(temp.recognize_google(audio))
