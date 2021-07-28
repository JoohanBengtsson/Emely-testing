general = {
    "MLI1TC1": "QA",
    "MLI4TC1": "QA",
    "MLI5TC1": "QA",
    "MLI6TC1": "IQA",
    "MLI7TC1": "IQA",
    "MLI13TC1": "CO",
    "MLU3TC1": "QA",
    "MLU4TC1": "QA",
    "MLU5TC1": "QA",
    "QA": 1000,
    "CO": 1100,
    "IQA": 1200,
    "n_CO": 2,
    "n_IQA": 4,
    "n_QA": 5
}

# Test 10xx: QA type (QA)
# Test 11xx: Consistency type. Ask for information (CO)
# Test 12xx: Indirect QA type (IQA)

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
                    "There is no sport better than football",
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
    "test": "QA",
    "id": 1200,
    "directed": False,
    "QA": "What is the name?",
    'answer': 'Johan',
    "information": ["I have a name. It is Johan",
                    "I have a name. I like that name. The name is Johan"],
    "question": ["What is my name?"]
}

ds1201 = {
    "test": "QA",
    "id": 1201,
    "directed": False,
    "QA": "What is the name?",
    "answer": "Mittens",
    "information": ["I have a cat. Its name is Mittens"],
    "question": ["What is the name of my cat?"]
}

ds1202 = {
    "test": "QA",
    "id": 1202,
    "directed": True,
    "QA": None,
    "answer": True,
    "information": ["I have a pet. It is a cat"],
    "question": ["Do I have any cats?"]
}

ds1203 = {
    "test": "QA",
    "id": 1203,
    "directed": True,
    "QA": None,
    "answer": False,
    "information": ["I have a pet. It is a cat"],
    "question": ["Do I have any dogs?"]
}