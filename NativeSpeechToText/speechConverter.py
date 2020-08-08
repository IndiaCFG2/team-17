import speech_recognition as sr
import os
from googletrans import Translator
import subprocess
from os import path
from pydub import AudioSegment
def main():
    # x=path.join(os.getcwd(),'welcome_te.mp3')
    #subprocess.call(['ffmpeg','-i','welcome_te.mp3','welcome_te.wav'])
    s=os.getcwd()
    #print(s)
    # print(x)
    sound =AudioSegment.from_file(r"..\sample_recordings\sample-rec-telugu.mp3")
    sound.export("sample-rec-telugu.wav",format="wav")
    audio="sample-rec-telugu.wav"
    r=sr.Recognizer()
    with sr.AudioFile(audio) as source:
        audio1=r.record(source)
        text=r.recognize_google(audio1)
        speech_trans=Translator()
        translated_output=speech_trans.translate(text,dest='en')
        print(translated_output.text)
        output=translated_output.text
    f=open("sample-rec-telugu.txt",'w+')
    f.write(output)
    f.close()

if __name__=="__main__":
    main()