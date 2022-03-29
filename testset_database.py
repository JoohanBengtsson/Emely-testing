general = {
    "TC_REQ_I5": "QA",
    "TC_REQ_I8": "QA",
    "TC_REQ_I9": "IA",
    "TC_REQ_I10": "QA",
    "TC_REQ_I11": "IQ",
    "TC_REQ_I1": "CO",
    "TC_REQ_U3": "QA",
    "TC_REQ_U4": "QA",
    "TC_REQ_U5": "QA",
    "TC_REQ_U6": "QA",
    "QA": 1000,
    "CO": 1100,
    "IA": 1200,
    "IQ": 1300,
    "n_QA": 5,
    "n_CO": 2,
    "n_IA": 4,
    "n_IQ": 4
}

# Test 10xx: QA type (QA)
# Test 11xx: Consistency type. Ask for information (CO)
# Test 12xx: Indirect answer type (IA)
# Test 13xx: Indirect question type (IQ)

ds1000 = {
    "test": "QA",
    "id": 1000,
    "directed": False,
    "QA": "What is the name?",
    'answer': 'Johan',
    "information": ["My name is Johan",
                    "I am Johan",
                    "You can call me Johan"],
    "question": ["What is my name?",
                 "What am I called?",
                 "Which name do I have?"]
}

ds1001 = {
    "test": "QA",
    "id": 1001,
    "directed": False,
    "QA": "What is the name?",
    "answer": "Mittens",
    "information": ["My cat is named Mittens",
                    "My cat is called Mittens",
                    "You can call my cat Mittens"],
    "question": ["What is the name of my cat?",
                 "What is my cat called",
                 "Which name does my cat have?"]
}

ds1002 = {
    "test": "QA",
    "id": 1002,
    "directed": True,
    "QA": None,
    "answer": True,
    "information": ["I have a cat named Mittens",
                    "My cat is called Mittens",
                    "I have two cats"],
    "question": ["Do I have any pets?",
                 "Have I got any pets?",
                 "Do I have any animals?",
                 "Are pets anything I have?",
                 "I have a nice car. Do I have any pets?"]
}

ds1003 = {
    "test": "QA",
    "id": 1003,
    "directed": False,
    "QA": "What is the weather?",
    'answer': 'Sunny',
    "information": ["It is sunny today",
                    "The sun is shining today",
                    "It is not raining today, but it is sunny"],
    "question": ["What is the weather like today?",
                 "What weather type is it today?"]
}

ds1004 = {
    "test": "QA",
    "id": 1004,
    "directed": False,
    "QA": "What is the favorite sport?",
    'answer': 'Football',
    "information": ["My favorite sport is football",
                    "There is no better sport than football",
                    "I love football, it is clearly my favorite sport"],
    "question": ["What is my favorite sport?",
                 "Which sport is my favorite?"]
}

ds1100 = {
    "test": "CO",
    "id": 1100,
    "directed": True,
    "QA": None,
    "information": ["Do you have any pets?",
                    "Have you got any pets?",
                    "Do you have any animals?",
                    "Are pets anything you have?",
                    "I love my cat. His name is Mittens. Do you have any pets?"],
    "question": None
}

ds1101 = {
    "test": "CO",
    "id": 1101,
    "directed": False,
    "QA": "What is the name?",
    "information": ["What is the name of your dog?",
                    "What is your dog called?",
                    "What have you named your dog?",
                    "Which name does your dog have?",
                    "I love my dog. What is the name of your dog?"],
    "question": None
}

ds1200 = {
    "test": "IA",
    "id": 1200,
    "directed": False,
    "QA": "What is the name?",
    'answer': 'Johan',
    "information": ["I have a name. It is Johan",
                    "I have a name. I like that name. The name is Johan",
                    "Johan is a nice name. It is my name actually"],
    "question": ["What is my name?"]
}

ds1201 = {
    "test": "IA",
    "id": 1201,
    "directed": False,
    "QA": "What is the name?",
    "answer": "Mittens",
    "information": ["I have a cat. Its name is Mittens",
                    "Mittens is a great name. It is actually the name of my cat."],
    "question": ["What is the name of my cat?"]
}

ds1202 = {
    "test": "IA",
    "id": 1202,
    "directed": True,
    "QA": None,
    "answer": True,
    "information": ["I have a pet. It is a cat"],
    "question": ["Do I have any cats?"]
}

ds1203 = {
    "test": "IA",
    "id": 1203,
    "directed": True,
    "QA": None,
    "answer": False,
    "information": ["I have a pet. It is a dog",
                    "I like pets. I have one myself. It is a dog"],
    "question": ["Do I have any dogs?"]
}

ds1300 = {
    "test": "IQ",
    "id": 1300,
    "directed": False,
    "QA": "What is the name?",
    'answer': 'Johan',
    "information": ["My name is Johan"],
    "question": ["Everyone has a name. What is mine?",
                 "I really like my name. What is it?"]
}

ds1301 = {
    "test": "IQ",
    "id": 1301,
    "directed": False,
    "QA": "What is the name?",
    "answer": "Mittens",
    "information": ["I have a cat named Mittens"],
    "question": ["I have a cat, as you know. What is its name?",
                 "I own one cat which I love playing with. What is its name?"]
}

ds1302 = {
    "test": "IQ",
    "id": 1302,
    "directed": True,
    "QA": None,
    "answer": True,
    "information": ["I have a cat"],
    "question": ["I like cats a lot. Do I have one?",
                 "Pets are awesome. Do I have one?"]
}

ds1303 = {
    "test": "IQ",
    "id": 1303,
    "directed": True,
    "QA": None,
    "answer": False,
    "information": ["I have cat"],
    "question": ["I like dogs a lot. Do I have one?",
                 "Dogs are awesome. Do I have one?"]
}
