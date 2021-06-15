import requests
import os
import random
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration, BlenderbotConfig
import pandas as pd
import torch
from detoxify import Detoxify

isBlenderbot = True # True: Emely talks to blenderbot, False: Emely talks to self
convarray = ["Hey","Hey"] # Array with the conversation.

toxicity_matrix = []
# Useful variables
#model = Detoxify('original', device='cuda')
# to specify the device the Detoxify-model will be allocated on (defaults to cpu), accepts any torch.device input

# Prints every row of the toxicity matrix, consists of the sentence + the different toxic aspects with their levels
def present_toxicities():
        print(row)
    for row in toxicity_matrix:
# Method for assessing the toxicity-levels of any text input, a text-array of any size


def analyze_word(text):
    # Each model takes in either a string or a list of strings
    #if len(text) == 1:
    #    results = Detoxify('original').predict(text)
        # Plain assessment of one string
    #else:
        # Assessment of several strings
    results = Detoxify('unbiased').predict(text)

    # Assessment of strings in multiple languages (probably not useful).
    #results = Detoxify('multilingual').predict(
    #     'пример текста'])
    #    ['example text', 'exemple de texte', 'texto de ejemplo', 'testo di esempio', 'texto de exemplo', 'örnek metin',

    # Presents the data as a Panda-Dataframe
    data_frame = pd.DataFrame(data=results, index=[text]).round(5)
    print(data_frame)
    toxicity_matrix.append(data_frame)


def array2string(convarray):
    # Converts the conversation array to a string separated by newline
    convstring = ' '.join([str(elem) + '\n' for elem in convarray])
    convstring = convstring[:len(convstring)-1]
    return convstring


def add2conversation(convarray, resp):
    # Adds a response and manages the amount of opening lines.
    convarray.append(resp)
    if i % 2 == 0:
        convarray.insert(0, "Hey")
    else:
        convarray.pop(0)

class blenderbot:
    def __init__(self):
        self.name = 'facebook/blenderbot-400M-distill'
        self.model = BlenderbotForConditionalGeneration.from_pretrained(self.name)
        self.tokenizer = BlenderbotTokenizer.from_pretrained(self.name)

    def getResponse(self, convarray):
        convstring = self.__array2blenderstring(convarray)
        inputs = self.tokenizer([convstring], return_tensors='pt')
        reply_ids = self.model.generate(**inputs)
        resp = self.tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0]
        return resp

    def __array2blenderstring(self, convarray):
        convstring = ' '.join([str(elem) + '</s> <s>' for elem in convarray])
        convstring = convstring[:len(convstring) - 8]
        return convstring

class emely:
    def __init__(self):
        self.URL = "http://localhost:8080/inference"

    def getResponse(self, convarray):
        # Inputs the conversation array and outputs a response from Emely
        jsonObj = {
            "accept": "application/json",
            "Content-Type": "application/json",
            "text": array2string(convarray)
        }
        r = requests.post(self.URL, json=jsonObj)
        resp = r.json()['text']
        return resp

if __name__ == '__main__':
    # If you want to start the chatbot
    #os.system("docker run -p 8080:8080 emely-interview")

    model_emely = emely()

    if isBlenderbot:
        model_blenderbot = blenderbot()

    # Loop a conversation
    for i in range(10):

        # Get response from the Emely model
        resp = model_emely.getResponse(convarray)
        print("Emely: ", resp)

        # Get next response.
        if isBlenderbot:
            convarray.append(resp)
            resp = model_blenderbot.getResponse(convarray[-3:])
            convarray.append(resp)
            print("Human: ", resp)
        else:
            print("Emely 2: ", resp)
            resp = add2conversation(convarray,resp)
        # Analyzes Emely's response and stores the assessment in a matrix. Send string as a matrix
        analyze_word([resp])


    # Save the entire conversation
    convstring = array2string(convarray)

    present_toxicities()
    # The method for presenting the toxicity levels per sentence used by Emely
