from googletrans import Translator
import csv

#input_file=open("sample_input_anylanguagetoenglish.txt","r", encoding="utf8")
#output_file=open("sample_output_anylanguagetoenglish.txt","a", encoding="utf8")
def anylangto_eng(i):
    """ input_file = []
    input_file.append(s)
    for row in input_file:
        col=row.split(",")
        temp=""
        for i in col: """
    speech_trans=Translator()
    temp=""
    translated_output=speech_trans.translate(i,dest='en')
    temp=temp+translated_output.text
    return temp

        #output_file.write(temp)
        #output_file.write("\n")
    
print("I'm Done!")
#input_file.close()
#output_file.close()
