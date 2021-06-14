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
    jsonText = conversation[0]

    # The JSON-object that should be sent as a parameter in the API-call to Emely.
    jsonObj = {
        "accept": "application/json",
        "Content-Type": "application/json",
        "text": jsonText
    }

    # Making the API POST-call
    r = requests.post(URL, json=jsonObj)

    # Accessing Emely's response and printing it
    resp = r.json()['text']
    print(resp)


