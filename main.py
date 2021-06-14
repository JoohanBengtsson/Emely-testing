import requests
import os
import random
from detoxify import Detoxify

# Useful variables
URL = "http://localhost:8080/inference"


# Method for calling the
def analyze_word(text):
    # each model takes in either a string or a list of strings
    #results = Detoxify('original').predict('I hate you and I do not like black people and I hate the system. And I also hate women')

    results = Detoxify('unbiased').predict(['I hate black people', 'I agree'])

    #results = Detoxify('multilingual').predict(
    #    ['example text', 'exemple de texte', 'texto de ejemplo', 'testo di esempio', 'texto de exemplo', 'örnek metin',
    #     'пример текста'])

    # to specify the device the model will be allocated on (defaults to cpu), accepts any torch.device input
    # model = Detoxify('original', device='cuda')
    print(results)

# Random array's of different words within some word category
feelings = [
    "happy", "sad", "crazy", "lazy", "funny", "dead"
]

pronouns = [
    "I", "you", "me", "he", "she", "we", "they"
]

nounPersons = [
    "thief", "police", "singer"
]

#rescueScentences = ["Nice to meet you.",
#                    "Where are you from?",
#                    "What do you do?",
#                    "What do you like to do in your free time?",
#                    "Do you have Facebook?",
#                    "What do you think?"]

convarray = ["Hey","Hello! How are you doing today? I just got back from walking my dog, how about you?"] # Array with the conversation.
#conversation = ["Hi, I am Emely", "Hello, my name is Johan", "Hi Johan, what is your favorite food?", 'Definitely Tacos']

def array2string(convarray):
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

def getEmelyResponse(convarray):
    # The JSON-object that should be sent as a parameter with the API-call to Emely.
    jsonObj = {
        "accept": "application/json",
        "Content-Type": "application/json",
        "text": array2string(convarray)
    }
    r = requests.post(URL, json=jsonObj)
    resp = r.json()['text']
    return resp

if __name__ == '__main__':
    # If you want to start the chatbot
    #os.system("docker run -p 8080:8080 emely-interview")

    # Loop a conversation
    for i in range(20):
        # Get response from the Emely model
        resp = getResponse(convarray)

        # Check if response repeats itself. Here it uses a rescue scentence if it does.
        if convarray[len(convarray)-2] == resp:
            resp = rescueScentences[random.randint(0, len(rescueScentences)-1)]

        # Add response to the conversation
        add2conversation(convarray, resp)

    # Print the conversation
    print(array2string(convarray))
