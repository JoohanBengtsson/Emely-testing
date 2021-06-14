import requests

URL = "http://localhost:8080/inference"

feelings = [
    "happy", "sad", "crazy", "lazy", "funny", "dead"
]

pronouns = [
    "I", "you", "me", "he", "she", "we", "they"
]

nounPersons = [
    "thief", "police", "singer"
]

conversation = ["Hi, I am Emely", "Hello, my name is Johan", "Hi Johan, what is your favorite food?", 'Definitely Tacos']


if __name__ == '__main__':
    for i in range(len(conversation)):
        jsonText = conversation[0]
        for sentenceIndex in range(1, len(conversation)):
            jsonText = jsonText + '\n' + conversation[sentenceIndex]
        jsonObj = {
            "accept": "application/json",
            "Content-Type": "application/json",
            "text": "Hey\n How are you doing?"
        }
        r = requests.post(URL, json=jsonObj)
        resp = r.json()['text']
        print(resp)


