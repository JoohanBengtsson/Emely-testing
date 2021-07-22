general = {
    "MLI4TC1": "QA",
    "MLI13TC1": "CO",
    "QA": 1000,
    "CO": 1100,
    "n_QA": 2,
    "n_CO": 2

}

# Test 10xx: QA type
# Test 11xx: Consistency type. Ask for information

ds1000 = {
    "test": "QA",
    "id": 1000,
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
    'answer': 'Mittens',
    "information": ["My cat is named Mittens",
                    "My cat is called Mittens",
                    "You can call my cat Mittens"],
    "question": ["What is the name of my cat?",
                 "What is my cat called",
                 "Which name does my cat have?"]
}

ds1100 = {
    "test": "CO",
    "id": 1100,
    "directed": True,
    "question": None,
    "words": ["Do you have any pets?",
              "Have you got any pets?",
              "Do you have any animals?",
              "Are pets anything you have?",
              "I love my cat. His name is Mittens. Do you have any pets?"]
}

ds1101 = {
    "test": "CO",
    "id": 1101,
    "directed": False,
    "question": "What is the name?",
    "words": ["What is the name of your dog?",
              "What is your dog called?",
              "What have you named your dog?",
              "Which name does your dog have?",
              "I love my dog. What is the name of your dog?"]
}
