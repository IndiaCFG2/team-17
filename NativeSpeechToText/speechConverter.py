import speech_recognition as sr
import os
import subprocess
from os import path
from pydub import AudioSegment
def main():
    # x=path.join(os.getcwd(),'welcome_te.mp3')
    #subprocess.call(['ffmpeg','-i','welcome_te.mp3','welcome_te.wav'])
    s=os.getcwd()
    #print(s)
    # print(x)
    sound =AudioSegment.from_file(r"..\sample_recordings\sample_rec_hindi.m4a")
    sound.export("sample_rec_hindi1.wav",format="wav")
    audio="sample_rec_hindi1.wav"
    r=sr.Recognizer()
    with sr.AudioFile(audio) as source:
        audio1=r.record(source)
        text=r.recognize_google(audio1)
        print(text)
    f=open("sample_rec_hindi.txt",'w+')
    f.write(text)
    f.close()

if __name__=="__main__":
    main()