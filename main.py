import requests
import os
import random
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration, BlenderbotConfig

isBlenderbot = True # True: Emely talks to blenderbot, False: Emely talks to self
convarray = ["Hey","Hey"] # Array with the conversation.

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

    # Save the entire conversation
    convstring = array2string(convarray)
