general = {
    "MLI1TC1": "QA",
    "MLI4TC1": "QA",
    "MLI5TC1": "QA",
    "MLI13TC1": "CO",
    "QA": 1000,
    "CO": 1100,
    "n_QA": 3,
    "n_CO": 2

}

# Test 10xx: QA type
# Test 11xx: Consistency type. Ask for information

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
