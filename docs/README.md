# 1. Introduction

This project aims to provide an open-source test framework that could be used to test any chatbot. The script is setup so that it is easy to add a new chatbot, test dataset or test case in order to assess it, more details about this can be found in **Implementation details**.

The script will produce a conversation between two chatters, hereafter called chatter1 respectively chatter2, and then assess the conversation with regards to some predefined quality aspects. The quality aspects will be defined below in the **2. Requirements evaluated by this testing framework** chapter and the test architecture is presented in the **3 Test framework** chapter. These quality aspects will be assessed and then written to a .xlsx-file, for the user to use for further assessment of the chatbot.

# 2. Requirements evaluated by this testing framework

The framework is evaluating a set of requirements within four different categories:
* Understanding - The understanding of the input itself.
* Intelligence  - The understandong of the context and the conversation.
* Personality   - The choice of answers.
* Answering     - The correctness of the answer.

## 2.1 Explanation of requirements and test cases

The codes within parentheses refer to the WIP paper.

| Category      | Code   | Explanation                                                            | Example                                | Test case category        |
|---------------|--------|------------------------------------------------------------------------|----------------------------------------|---------------------------|
| Understanding | ML-U3 (U3) | Understanding of sentences with typing mistakes                        | Whqt is tje name rf my cat?            | Question-Answer           |
|               | ML-U4 (U4) | Understanding of sentences with incorrect word order                   | What the is name of cat my?            | Question-Answer           |
|               | ML-U5 (U5)  | Understanding of sentences with some words left out                    | What the name my cat?                  | Question-Answer           |
|               | ML-U6 (U6) | Understanding of sentences with some words replaced with a random word | What banana the name of my cat?        | Question-Answer           |
| Intelligence  | ML-I1 (I5)  | Long term memory assessment                                            |                                        | Question-Answer           |
|               | ML-I2 (I2) | Coherence assessment wrt the conversation                              | “I like cars”, “That sounds tasty”     | Sentence-BERT             |
|               | ML-I3 (I3) | Coherence assessment wrt the input sentence                            | The kitten which I own is called John. | Question-Answer           |
|               | ML-I4 (I8) | Different formulated information assessment                            | The kitten which I own is called John. | Question-Answer           |
|               | ML-I5 (I10)  | Different formulated questions assessment                              | What name does my cat possess?         | Question-Answer           |
|               | ML-I6 (I9) | Context dependent information understanding                            | I have a cat. His name is John.        | Question-Answer           |
|               | ML-I7 (I11) | Context dependent questions understanding                              | I love my cat. What was its name?      | Question-Answer           |
|               | ML-I13 (I1) | Consistency with own information                                       |                                        | Question                  |
| Personality   | ML-P1 (P2) | Toxicities assessment                                                  |                                        | Detoxifyer                |
| Answering     | ML-A6 (A4) | Repetition avoidance assessment                                        | I have I have I have a cat.            | N-grams                   |
|               | ML-A7 (A3) | Repeated questions avoidance assessment                                |                                        | Simple check              |

* Understanding of sentences with typing mistakes (ML-U3):
  How successful the chatbot is to understand sentences with words having typing mistakes. It is given some information and then asked about the information with typing mistakes. The typing mistake introduction only applies on the questions. The amount of words having typing mistakes and the proportion of typing mistakes in each of these words is randomized for each test.
  The presented score is a primary list containing secondary lists with format [successful attempts, total number of attempts]. Each element of the primary list is related to the (number of typing mistake words * proportion typing mistakes). It gives a base for creating a histogram over success rate over the number of swaps.
  The detailed information (optional) is the assessed sentence similarity between the chatbot's answer and the correct answer.
  The interpretation (optional) is the interpretation of the answer from the chatbot.

* Understanding of sentences with incorrect word order (ML-U4):
  How successful the chatbot is to understand sentences with words presented in an incorrect order. It is given some information and then asked about the information with words swapped. The word swap only applies on the questions. The amount of words swapped is randomized for each test.
  The presented score is a primary list containing secondary lists with format [successful attempts, total number of attempts]. Each element of the primary list is related to the number of swaps in a sentence. The format is therefore [[success 0 swaps, total 0 swaps], [success 1 swaps, total 1 swaps], [success 2 swaps, total 2 swaps], ...]. It gives a base for creating a histogram over success rate over the number of swaps.
  The detailed information (optional) is the assessed sentence similarity between the chatbot's answer and the correct answer.
  The interpretation (optional) is the interpretation of the answer from the chatbot.

* Understanding of sentences with some words left out (ML-U5):
  How successful the chatbot is to understand sentences with some words left out. It is given some information and then asked about the information with words left out. The word masking only applies on the questions. The amount of words swapped is randomized for each test.
  The presented score is a primary list containing secondary lists with format [successful attempts, total number of attempts]. Each element of the primary list is related to the proportion of words left out in a sentence. The format is therefore [[success 0-5% removed, total 0-5% removed], [success 5-10% removed, total 5-10%removed], [success 10-15% removed, total 10-15% removed], ...]. It gives a base for creating a histogram over success rate over the number of words removed.
  The detailed information (optional) is the assessed sentence similarity between the chatbot's answer and the correct answer.
  The interpretation (optional) is the interpretation of the answer from the chatbot.

* Understanding of sentences with some words replaced with a random word (ML-U6):
  How successful the chatbot is to understand sentences with some words replaced with a random word. It is given some information and then asked about the information with words replaced. The word replacement only applies on the questions. The amount of words replaced is randomized for each test.
  The presented score is a primary list containing secondary lists with format [successful attempts, total number of attempts]. Each element of the primary list is related to the proportion of words replaced in a sentence. The format is therefore [[success 0-5% replaced, total 0-5% replaced], [success 5-10% replaced, total 5-10% replaced], [success 10-15% replaced, total 10-15% replaced], ...]. It gives a base for creating a histogram over success rate over the number of words replaced.
  The detailed information (optional) is the assessed sentence similarity between the chatbot's answer and the correct answer.
  The interpretation (optional) is the interpretation of the answer from the chatbot.

* Long term memory assessment (ML-I1) - how successful the chatbot is to remember details after varying amount of conversation rounds.
  The presented score is a list with format [successful attempts, total number of attempts].
  The detailed information (optional) is the assessed sentence similarity between the chatbot's answer and the correct answer.
  The interpretation (optional) is the interpretation of the answer from the chatbot.

* Coherence assessment wrt the conversation (ML-I2):
  Assesses if the response produced by a chatter coherent with respect to the whole conversation**. Assessed with Sentence-BERT.
  The presented score is a list with format [successful attempts, total number of attempts].
  The detailed information (optional) is the assessed sentence similarity between the chatbot's answer and the correct answer.
  The interpretation (optional) is the interpretation of the answer from the chatbot.

* Coherence assessment wrt the input sentence (ML-I3):
  Assesses if the response produced by a chatter coherent with respect to the input sentence. Assessed with Sentence-BERT.
  The presented score is a list with format [successful attempts, total number of attempts].
  The detailed information (optional) is the assessed sentence similarity between the chatbot's answer and the correct answer.
  The interpretation (optional) is the interpretation of the answer from the chatbot.

* Different formulated information assessment (ML-I4):
  How successful the chatbot is to understand different formulated information. It is first given some information formulated in different ways and then asked about the information.
  The presented score is a list with format [successful attempts, total number of attempts].
  The detailed information (optional) is the assessed sentence similarity between the chatbot's answer and the correct answer.
  The interpretation (optional) is the interpretation of the answer from the chatbot.

* Different formulated questions assessment (ML-I5):
  How successful the chatbot is to understand different formulated questions. It is first given some information and then asked about the information with questions formulated in different ways.
  The presented score is a list with format [successful attempts, total number of attempts].
  The detailed information (optional) is the assessed sentence similarity between the chatbot's answer and the correct answer.
  The interpretation (optional) is the interpretation of the answer from the chatbot.

* Context dependent information understanding (ML-I6):
  How successful the chatbot is to understand information that requires context. It is first given some context in one sentence and then the information in one sentence. For example "I like my cat. His name is Mittens". It is then asked about the information.
  The presented score is a list with format [successful attempts, total number of attempts].
  The detailed information (optional) is the assessed sentence similarity between the chatbot's answer and the correct answer.
  The interpretation (optional) is the interpretation of the answer from the chatbot.

* Context dependent questions understanding (ML-I7):
  How successful the chatbot is to understand questions that requires context. It is first given some information. It is then asked about the information in a context dependent way. For example "I like my cat. What is its name?".
  The presented score is a list with format [successful attempts, total number of attempts].
  The detailed information (optional) is the assessed sentence similarity between the chatbot's answer and the correct answer.
  The interpretation (optional) is the interpretation of the answer from the chatbot.

* Consistency with own information (ML-I13):
  How consistent the chatbot is when asked about its own information. It is then asked about its own information several times and the similarity of the answers are assessed. The correct answer is set to the first answer of the chatbot.
  The presented score is a list with format [successful attempts, total number of attempts].
  The detailed information (optional) is the assessed sentence similarity between the chatbot's answer and the correct answer.
  The interpretation (optional) is the interpretation of the answer from the chatbot.

* Toxicities assessment (ML-P1)
  different kinds of toxicities are assessed. Test passes if the value of the certain test variable is below a certain threshold.
  The presented score is a list with format [successful attempts, total number of attempts].
  The detailed information (optional) is the assessed %-risk of the response being interpreted as the specific toxicity type.

* Repetition avoidance assessment (ML-A6):
  How successful is the chatbot at avoiding unnecessary repeating of sentences or words. Assessed using using Ngrams. It is created by taking the mean of the N-gram order for N = 2:max times the amount of times that N-gram occurs, mean(order*occurrences of that order) for order 2:max. A 2-gram occurring six times adds the same as a 6-gram occurring 2 times. The test passes if the stuttering value is below a certain threshold.
  The presented score is a list with format [successful attempts, total number of attempts].
  The detailed value is the assessed stuttering value.

* Repeated questions avoidance assessment (ML-A7)
   Assesment if the chatbot repeats any question several times.
   The presented score is a list with format [successful attempts, total number of attempts].

** The whole conversation is not brought as input, since chatbots have varying sizes of input they are able to handle. Instead, a function is implemented to bring the last sentences consisting a maximum of X tokens, in order to handle this.

# 3 Test framework
## 3.1 Framework contents
### 3.1.1 files
* main.py                   - contains the main method and central functions in the test architecture.
* test_functions.py         - contains the functions which are used to analyse the test responses.
* util_functions.py         - contains utility functions which can be of help anywhere in the code.
* config.py                 - configuration document. Contains all settings. Only document from where a user should change the code.
* testset_database.py       - contains the dataset of all tests which are used.
* validation_of_metrics.py  - only used when validating accuracy and finding thresholds.

### 3.1.2 external models
* BERT-SQuAD
  * Input: A string with information, a string with some question about the information.
  * Output: The answer to the question. (string)
* SentenceTransformer
  * Input: Two strings.
  * Output: Semantic similarity between the two strings (float between 0 and 1).
* Detoxifyer
  * input: Some string.
  * Output: The toxicity of the string based on seven measurements:
    * toxicity        (float between 0 and 1)
    * severe toxicity (float between 0 and 1)
    * obscene         (float between 0 and 1)
    * identity attack (float between 0 and 1)
    * insult          (float between 0 and 1)
    * threat          (float between 0 and 1)
    * sexual_explicit (float between 0 and 1)
* Sentence-BERT
  * input: Two strings
  * output: The coherence of the second string given the first string (float between around -12 and 12).

## 3.2 Test architecture

1. The function init_tests() is called. It allocates which tests should be performed and where, based on the configurations. The output is the test allocation list test_ids and the corresponding test sets in test_sets.
2. The function generate_conversation() is called, which generates the conversation and injects tests based on the test_ids array. The output is the generated conversation.
3. The function analyze_conversation() is called, which assesses the answers. The output is a data frame with results for each test. The summary of the entire run is added in the bottom and exported to the results summary document.
4. The test is repeated until a specified amount if runs are finished. The summary of the entire results summary document is added in the bottom.

![Image of test architecture](https://github.com/JoohanBengtsson/Emely-testing/blob/main/images/Architecture.png)

## 3.3 Test case categories

There are four test case categories used.
* Question-Answer
* Question
* N-grams
* Sentence-BERT
* Detoxifyer
* Simple check

### 3.3.1 Question-Answer
In Question-Answer test, the tester first gives some information and then asks a question and demands an answer about the information. If the answer is correct, the test has passed. There are two types of questions, one for open questions and one for closed (yes/no).

#### 3.3.1.1 Open questions
The algorithm for assessing open questions is shown below:
1. To understand whether the answer is correct, the answer is inserted into the BERT-SQuAD model along with a question specified in the test dataset to get the right information.
2. The output is a more condensed answer which is inserted into the SentenceTransformers model along with the correct answer specified in the dataset.
3. The output is the semantic similarity between the two answers, and is an interpretation of the correctness of the answer.
4. If the value is above a certain threshold, the test has passed.

![Image of QA model for open questions](https://github.com/JoohanBengtsson/Emely-testing/blob/main/images/QA-model.png)

#### 3.3.1.2 Closed questions
The algorithm for assessing closed questions is shown below:
1. The answer is scanned for the formulations "no", "don't" and "do not". If the answer contains any of these, the answer is labeled False (No). Otherwise, the answer is set to True (yes).
2. If the answer is the same as the correct one in the dataset, the test has passed.

### 3.3.2 Question
The Question test is the same as Question-Answer, with the exception that there is no predefined correct answer. Instead, the bot itself is asked for some information which is then considered the correct answer. The bot can thereafter be asked the same question again and the correctness of that answer can be assessed based on the previous answer.

### 3.3.3 N-grams
N-grams testing makes use of the structure of the answer instead of analysing the contents of the answer. It can therefore be used to assess if there are any trends or abnormalities in the structure of the dataset.

### 3.3.4 Sentence-BERT
Sentence-BERT test method makes direct use of the Sentence-BERT model to assess the coherence of the answer.

### 3.3.5 Detoxifyer
Detoxifyer test method makes direct use of the Detoxifyer model to assess the suitability of the answer.

### 3.3.5 Simple check
Simple check test method tests a simple property of the conversation.

## 3.4 Load conversation format
Loading a conversation takes in a conversation file in the load_conv_folder folder. In that folder, each text file will be loaded.
The format of the loaded text document is the following:
1. Each newline is a new reply.
1. If test cases are used and will be assessed, there must be the following components in the document:
  1. a "- CONFIGURATIONS -" row directly after the conversation.
  1. a test_idx list directly after the "- CONFIGURATIONS -" row surrounded by "test_idx" ("test_idx[0, 0, 1141001, 1141001.5, 0, 0]test_idx")
  1. a test_sets dictionary directly after the "test_idx" row
An example of the format is given below. Easiest way to get more examples of the format is to generate and save a conversation.
If only the conversation is in the text, only the tests which apply for each response will work.

```
hi , how are you today ?
hi , I'm good. And you ?
i ' m good , thank you .
what are your favorite things to do ?
You can call my cat Mittens
OK. Got it.
What is the name of my cat?
It's Mittens
Ok, good.
Nice to talk to you
- CONFIGURATIONS -
test_ids[0, 0, 1041001, 1041001.5, 0]test_ids
test_sets{'MLI4TC1': [{'test': 'QA', 'id': 1001, 'directed': False, 'QA': 'What is the name?', 'answer': 'Mittens', 'information': ['My cat is named Mittens', 'My cat is called Mittens', 'You can call my cat Mittens'], 'question': ['What is the name of my cat?', 'What is my cat called', 'Which name does my cat have?']}]}test_sets
```

# 4. Instructions

This section aims to guide the reader on how to clone, run  and use the script.

## 4.1 Clone and setup repository locally

It is a precondition that the user has the following parts ready on the computer:
* Python, with pip, virtual environments and an IDE

Whenever the preconditions are fulfilled, the user may start using the script. In order to do so, the user needs to clone the repo. That is done either through the command prompt or through a tool for source control. Here, the command prompt-way is demonstrated:

1. Start the command prompt
2. Navigate to wherever you want to place the local repository
3. Make the command:

``git clone https://github.com/JoohanBengtsson/Emely-testing.git``

4. To setup the virtual environment, follow the following steps:

* Navigate into the directory: ``cd Emely-testing``  
and then create the environment: ``python -m venv env``
* Activate the environment:
``env\Scripts\activate``
* Install the required packages: ``pip install -r requirements.txt``
* If your computer has a GPU and you have a **Windows**-computer, the following line will reinstall Pytorch with support for using the GPU: ``pip3 install torch==1.9.0+cu102 torchvision==0.10.0+cu102 torchaudio===0.9.0 -f https://download.pytorch.org/whl/torch_stable.html``, otherwise visit https://pytorch.org/get-started/locally/ for more information on how to install Pytorch with GPU-support.
* Start the script using the recently setup environment within your preferred IDE.
5. Prior to running the script, some setting variables need explanation, which can be found in *4.2 Setting variables*. Some variables marked with * need to be specified prior to running the script. After specifying these values, the script is ready to be run, at least with the on forehand implemented conversational agents. If the agent that should be run is not implemented, please check *4.4 Implementation details of a new chatter*.

## 4.2 Setting variables

In this subsection, all the current setting variables will be explained.

| Variable name                | Variable description                                                                                                                      |
|------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------|
| **GENERAL**                      |                                          |
| max_runs*                     | Decides how many conversations that should be done in total |
| is_load_conversation         | True = Load from load_document. False = Generate text from the chatters specified below. |
| is_save_conversation         | True = Save conversation in folder save_documents |
| is_analyze_conversation      | True = if the program shall print metrics to .xlsx. False = If it is not necessary |
| **GENERATE**                     |                                          |
| conversation_length*          | Decides how many responses the two chatters will contribute with |
| init_conv_randomly           | True if the conversation shall start randomly using external tools. If chatter is set to |
|                              | either 'predefined' or 'user', this is automatically set to False |
| chatters*                     | Chatter 1-profile is on index 0, chatter 2-profile is on index 1. Could be either one of ['emely', 'blenderbot', 'user', 'predefined'].
|                              | 'emely' assigns Emely to that chatter.
|                              |'blenderbot' assigns Blenderbot to that chatter.  |
|                              |'user' lets the user specify the answers. |
|                              | 'predefined' loops over the conversation below in the two arrays predefined_conv_chatter1 and predefined_conv_chatter2. Two standard conversation arrays setup for enabling hard-coded strings and conversations and try out the metrics. |
|          convarray_init                     |Array for storing the conversation. Can be initialized as ["Hey", "Hey"] etc |
| predefined_conv_chatter1 / predefined_conv_chatter2      | Predefined conversations as per chatter (the number corresponds to the specific chatter, where number 2 is the one being tested).                  |
| prev_conv_memory_chatter1 / prev_conv_memory_chatter2     | How many previous sentences in the conversation shall be brought as input to any chatter. Concretely = conversation memory per chatter, if it needs to be controlled for a chatter. |
| **AFFECTIVE TEXT GENERATION**    |                                          |
|# is_affect                  |   Whether or not the affective text generator should be activated for first sentence. False by default |
| affect                       | Affect for text generation. ['fear', 'joy', 'anger', 'sadness', 'anticipation', 'disgust', 'surprise', 'trust']                     |
| knob                         | Amplitude for text generation. 0 to 100  |
| topic                        | Topic for text generation. ['legal','military','monsters','politics','positive_words', 'religion','science','space','technology'] |
| **SAVE AND LOAD**                |                                          |
| save_conv_folder             | The folder in which the conversations are saved |
| load_conv_folder             | The folder in which the conversations are contained |
| save_analysis_name           | The name of the analysis folder          |
| **ANALYSIS***                    |                                          |
| QA_model                     | Can be ['pipeline', 'bert-squad']. Defaults to 'pipeline', indicating that only the QA-model from transformers using pipeline will be used. Somewhat worse performance, but is easier to setup. To use 'bert-squad', it needs to be setup according to 4.4 in the readme. |
| show_interpret               | Interpretations, whether the result should show how the script interprets the analyzed responses. True or False (ToF)                         |
| show_detailed                | Detailed results, that is the scores received from the used ML-models. ToF                        |
| show_binary                  | Binary results - whether an answer's detailed value passes or not a threshold value. ToF                          |
| is_analyze_question_freq     | Question frequency - ToF                       |
| is_MLP1TC1                   | Toxicity analysis using a toxicity analysis tool. ToF                                 |
| is_MLI2TC1                   | Context coherence, wrt the whole conversation. ToF                        |
| is_MLI3TC1                   | Sentence coherence, wrt last sentence. ToF                       |
| is_MLA6TC1                   | Stuttering. ToF                               |
| p_MLI1TC1                    | Remember information for a certain amount of time. Any float value in the range [0, 1]. |
| p_MLI4TC1                    | Understand differently formulated information. [0, 1] |
| p_MLI5TC1                    | Understand differently formulated questions. [0, 1] |
| p_MLI6TC1                    | Understand information based on context. [0, 1]  |
| p_MLI7TC1                    | Understand questions based on context. [0, 1]    |
| p_MLI13TC1                   | Consistency with own information. [0, 1]         |
| p_MLU3TC1                    | Understands questions with randomly inserted typing mistakes. [0, 1] |
| p_MLU4TC1                    | Understands questions with randomly swapped word order. [0, 1] |
| p_MLU5TC1                    | Understands questions with randomly masked words. [0, 1] |
| p_MLU6TC1                    | Understands questions with some words swapped for randomly chosen words. [0, 1] |
| **AUXILIARY ANALYSIS VARIABLES** |                                          |
| maxsets_MLI1TC1              | How many different data sets may be used for MLI1TC1. Depends on how many QA-data sets that are available, but the value should be in the range [1, 5]. |
| maxsets_MLI4TC1              | How many different data sets may be used for MLI4TC1. [1, 5] |
| maxsets_MLI5TC1              | How many different data sets may be used for MLI5TC1. [1, 5] |
| maxsets_MLI6TC1              | How many different data sets may be used for MLI6TC1.|
| maxsets_MLI7TC1              | How many different data sets may be used for MLI7TC1.|
| maxsets_MLI13TC1             | How many different data sets may be used for MLI13TC1.|
| maxsets_MLU3TC1              | How many different data sets may be used for MLU3TC1. [1, 5] |
| maxsets_MLU4TC1              | How many different data sets may be used for MLU4TC1. [1, 5] |
| maxsets_MLU5TC1              | How many different data sets may be used for MLU5TC1. [1, 5] |
| maxsets_MLU6TC1              | How many different data sets may be used for MLU6TC1. [1, 5] |
| maxlength_MLI1TC1            | Maximum amount of rounds that the ML1TC1 can wait for to test long term memory. Value should be in the range [1, conversation_length - 1]. |
| array_ux_test_cases          | The array consisting of the test cases related to understanding, in which it is relevant to store the results and map the results to different levels of inserted errors.              |
| threshold_sem_sim_tests      | The threshold used for the QA-models using semantic similarity. The threshold level is the threshold used for assessing the values received from the ML model. |
| **DATA AUGMENTATION**            |                                          |
| p_synonym                    | Probability of switching to a synonym    |
| n_aug                        | Number of times each test set should be augmented by switching some words with synonyms |

## 4.3 Setting up BERT-SQuAD
BERT-SQuAD can be setup in two different ways, and controlled by the QA_model variable. Selecting 'pipeline' does not require extra software and can be used right away. Selecting 'bert-squad' requires extra download according to the process:

1. Clone git repository ``git clone https://github.com/kamalkraj/BERT-SQuAD.git`` inside the root folder.
2. Install requirements ``pip install -r ./BERT-SQuAD/requirements.txt``
3. Download pretrained model from website https://www.dropbox.com/s/8jnulb2l4v7ikir/model.zip
4. Unzip and move the files inside ./BERT-SQuAD/model/

## 4.4 Implementation details of a new chatter

The script is currently offering four chatter profiles for generating the conversation, namely:
* Emely
* Blenderbot 400M
* User - enabling the user to interact with the other chatter
* Predefined sentences located in predefined_conv_chatter1 and predefined_conv_chatter2

These four chatter types have been implemented as classes, in the **Classes-section** within the code. These classes all have the method **get_response(self, conv_array)** in common. This method is what mainly differs between the chatter types on how text is generated. Hence, if the user wants to add another chatbot, it only takes to go through three steps:
1. Define a class and its own get_response(self, conv_array)-method, more specifically how to use the source for generating input.

```
class {BOT_CLASS_NAME}:
    def __init__(self):

    def get_response(self, convarray):
        response = {CODE FOR GENERATING RESPONSE FROM THE SOURCE}
        return response
```

2. Go to the method assign_model() and add:

```python
elif chatter_profile == {NAME_OF_BOT}:
    return {BOT_CLASS_NAME}()
```

3. In the config-script, go to the attribute array **chatters** and change either one or both of the indices 0 and 1 to {NAME_OF_BOT}
```python
chatters = ['emely', 'blenderbot']  # Chatter 1-profile is on index 0, chatter 2-profile is on index 1.
# Could be either one of ['emely', 'blenderbot', 'user', 'predefined']
```

Voilà: the framework is setup to include your robot as well and you are ready to test your robot.


## 4.5 Implementation details of a new dataset

A test dataset, or testset, is the dataset which is used for certain tests. The testset database is currently offering four different categories of test sets:
* Question-Answer (QA)
* Consistency (CO)
* Indirect answer (IA)
* Indirect question (IQ)

These categories have different formulations and are used for different tests. The structure is the same for all datasets. To add another test set, go through these steps:

1. Create the test set inside testset_database.
```
dsXXYY = {                                              # XX is the test category (see the general set inside testset_database), YY is the test in order.
    "test": "QA",                                       # The test type. "QA", "CO" etc
    "id": XXYY,                                         # The id of the test. Must be the same as XXYY in the name.
    "directed": False,                                  # Whether the question is directed or undirected. Directed is the same as closed, that is yes/no-question.     
    "QA": "What is the name?",                          # The question used in the SentenceTransformer model. Set to None if directed is true.
    'answer': 'Johan',                                  # The correct answer.
    "information": ["My name is Johan",                 # The information given to the tested bot with different formulations.
                    "I am Johan",
                    "You can call me Johan"],
    "question": ["What is my name?",                    # The question given to the tested bot with different formulations.
                 "What am I called?",
                 "Which name do I have?"]
}
```
2. Increase the number of datasets of the added type inside general in testset_database.

## 4.6 Implementation details of a new test case

A test case is a way of testing the bot. It is possible to add a new test case based on a new requirement in the Software Requirements Specification. It is also possible to add a test case based on a requirement that already has another test case.

To add another test case, do the following steps:
1. Decide the requirement and type of test.
  2. The format of the name is AABXTCY, where AA is the category of bot (ML=general bot, FK=fikakompis, AF=arbetsförmedling). So far, only ML has been used. B is the category of requirements (I = intelligence, P = personality etc). X is the number of the requirement. Same as in the Software Requirement Specification. TC stands for Test Case, and Y is the test case number. 1 is the first test of this requirement, 2 is the second etc.
  3. The index of the test is of a format XYY0000, where X is the requirement category (1 = intelligence, 2 = understanding, etc.) and YY is the number inside the category.
  4. The type of test depends on how the requirement should be tested. There are existing types of tests as shown in **4.3 Implementation details of a new dataset**.
5. If the test needs a test dataset, assign which type or dataset should be used. In testset_database, inside general, add the test name and type of test.
  6. if a new kind of test dataset is needed, add new datasets in the same way as in **4.3 Implementation details of a new dataset**.
7. Add the necessary parameters in the config file. Mandatory parameters are toggle for doing the test or not. If test datasets are used, maxsets i.e. the maximal number of separate sets in a test is mandatory.
8. Add the assign_dataset() function for the new test in main.init_test(), if the test requires a dataset.
9. Add the line of code inside main.init_test that makes it possible for the test case to be allocated inside the test_idx list.
10. Add the call for the test case inside main.analyze_conversation().
1.1 Add the test assessment itself inside the test_functions file. It should add the test results inside the data_frame and return it.

# 5. Software Requirements Specification

All software requirements for the chatbot can be shown in the System Requirements Specification, given upon request.

## 5.1 User stories

- Emely as the recruiter
- Emely as the fika buddy

# 6. Versions for working framework
absl-py==0.12.0
anyio==3.2.0
appnope @ file:///opt/concourse/worker/volumes/live/5f13e5b3-5355-4541-5fc3-f08850c73cf9/volume/appnope_1606859448618/work
argon2-cffi @ file:///opt/concourse/worker/volumes/live/d733ceb5-7f19-407b-7da7-a386540ab855/volume/argon2-cffi_1613037492998/work
astunparse==1.6.3
async-generator @ file:///home/ktietz/src/ci/async_generator_1611927993394/work
attrs @ file:///tmp/build/80754af9/attrs_1620827162558/work
Babel==2.9.1
backcall @ file:///home/ktietz/src/ci/backcall_1611930011877/work
bidict==0.21.2
bleach @ file:///tmp/build/80754af9/bleach_1612211392645/work
blis==0.7.4
boto3==1.17.112
botocore==1.20.112
cachetools==4.2.2
catalogue==2.0.4
certifi==2021.5.30
cffi @ file:///opt/concourse/worker/volumes/live/0ef369cc-6ba0-47e7-75da-208c6400381d/volume/cffi_1613246948181/work
chardet==3.0.4
click==7.1.2
collection==0.1.6
configparser==5.0.2
cycler==0.10.0
cymem==2.0.5
decorator @ file:///tmp/build/80754af9/decorator_1621259047763/work
defusedxml @ file:///tmp/build/80754af9/defusedxml_1615228127516/work
detoxify==0.2.2
en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.1.0/en_core_web_sm-3.1.0-py3-none-any.whl
entrypoints==0.3
et-xmlfile==1.1.0
filelock==3.0.12
Flask==2.0.1
Flask-Cors==3.0.10
Flask-SocketIO==5.1.0
flatbuffers==1.12
gast==0.4.0
google-auth==1.31.0
google-auth-oauthlib==0.4.4
google-pasta==0.2.0
googletrans==3.0.0
grpcio==1.34.1
h11==0.9.0
h2==3.2.0
h5py==3.1.0
hpack==3.0.0
hstspreload==2020.12.22
httpcore==0.9.1
httpx==0.13.3
huggingface-hub==0.0.8
hyperframe==5.2.0
idna==2.10
importlib-metadata @ file:///opt/concourse/worker/volumes/live/a634a87c-b5e5-41bd-628d-cd0413666c93/volume/importlib-metadata_1617877368300/work
IProgress==0.4
ipykernel @ file:///opt/concourse/worker/volumes/live/88f541d3-5a27-498f-7391-f2e50ca36560/volume/ipykernel_1596206680118/work/dist/ipykernel-5.3.4-py3-none-any.whl
ipython @ file:///opt/concourse/worker/volumes/live/c432d8a7-d8f3-4e24-590f-f03d7e5f35e1/volume/ipython_1617120884257/work
ipython-genutils @ file:///tmp/build/80754af9/ipython_genutils_1606773439826/work
ipywidgets==7.6.3
itsdangerous==2.0.1
jedi==0.17.0
Jinja2 @ file:///tmp/build/80754af9/jinja2_1621238361758/work
jmespath==0.10.0
joblib==1.0.1
json5==0.9.6
jsonschema @ file:///tmp/build/80754af9/jsonschema_1602607155483/work
jupyter==1.0.0
jupyter-client @ file:///tmp/build/80754af9/jupyter_client_1616770841739/work
jupyter-console==6.4.0
jupyter-core @ file:///opt/concourse/worker/volumes/live/c8df8dce-dbb3-46e7-649c-adf4ed2dd00a/volume/jupyter_core_1612213293829/work
jupyter-server==1.8.0
jupyterlab==3.0.16
jupyterlab-pygments @ file:///tmp/build/80754af9/jupyterlab_pygments_1601490720602/work
jupyterlab-server==2.6.0
jupyterlab-widgets==1.0.0
Keras==2.4.3
keras-nightly==2.5.0.dev2021032900
Keras-Preprocessing==1.1.2
kiwisolver==1.3.1
lxml==4.6.3
Markdown==3.3.4
MarkupSafe @ file:///opt/concourse/worker/volumes/live/c9141381-1dba-485b-7c96-99007bf7bcfd/volume/markupsafe_1621528150226/work
matplotlib==3.4.2
mistune @ file:///opt/concourse/worker/volumes/live/95802d64-d39c-491b-74ce-b9326880ca54/volume/mistune_1594373201816/work
multitasking==0.0.9
murmurhash==1.0.5
nbclassic==0.3.1
nbclient @ file:///tmp/build/80754af9/nbclient_1614364831625/work
nbconvert @ file:///opt/concourse/worker/volumes/live/2b9c1d93-d0fd-432f-7d93-66c93d81b614/volume/nbconvert_1601914875037/work
nbformat @ file:///tmp/build/80754af9/nbformat_1617383369282/work
nest-asyncio @ file:///tmp/build/80754af9/nest-asyncio_1613680548246/work
nltk==3.6.2
notebook @ file:///opt/concourse/worker/volumes/live/78fd3e35-67c2-490e-7bb9-0627a6db9485/volume/notebook_1621528340294/work
numpy==1.19.5
oauthlib==3.1.1
openpyxl==3.0.7
opt-einsum==3.3.0
packaging @ file:///tmp/build/80754af9/packaging_1611952188834/work
pandas==1.2.4
pandocfilters @ file:///opt/concourse/worker/volumes/live/c330e404-216d-466b-5327-8ce8fe854d3a/volume/pandocfilters_1605120442288/work
parso @ file:///tmp/build/80754af9/parso_1617223946239/work
pathy==0.6.0
patsy==0.5.1
pexpect @ file:///tmp/build/80754af9/pexpect_1605563209008/work
pickleshare @ file:///tmp/build/80754af9/pickleshare_1606932040724/work
Pillow==8.2.0
preshed==3.0.5
prometheus-client @ file:///tmp/build/80754af9/prometheus_client_1623189609245/work
prompt-toolkit @ file:///tmp/build/80754af9/prompt-toolkit_1616415428029/work
protobuf==3.17.3
ptyprocess @ file:///tmp/build/80754af9/ptyprocess_1609355006118/work/dist/ptyprocess-0.7.0-py2.py3-none-any.whl
pyasn1==0.4.8
pyasn1-modules==0.2.8
pycparser @ file:///tmp/build/80754af9/pycparser_1594388511720/work
pydantic==1.8.2
Pygments @ file:///tmp/build/80754af9/pygments_1621606182707/work
pyparsing @ file:///home/linux1/recipes/ci/pyparsing_1610983426697/work
pyrsistent @ file:///opt/concourse/worker/volumes/live/ff11f3f0-615b-4508-471d-4d9f19fa6657/volume/pyrsistent_1600141727281/work
python-dateutil @ file:///home/ktietz/src/ci/python-dateutil_1611928101742/work
python-engineio==4.2.0
python-socketio==5.3.0
pytorch-transformers==1.2.0
pytz==2021.1
PyYAML==5.4.1
pyzmq==20.0.0
qtconsole==5.1.0
QtPy==1.9.0
regex==2021.4.4
requests==2.25.1
requests-oauthlib==1.3.0
rfc3986==1.5.0
rsa==4.7.2
s3transfer==0.4.2
sacremoses==0.0.45
scikit-learn==0.24.2
scipy==1.6.3
Send2Trash @ file:///tmp/build/80754af9/send2trash_1607525499227/work
sentence-transformers==2.0.0
sentencepiece==0.1.95
six @ file:///tmp/build/80754af9/six_1623709665295/work
smart-open==5.1.0
sniffio==1.2.0
spacy-legacy==3.0.8
srsly==2.4.1
statsmodels==0.12.2
tensorboard==2.5.0
tensorboard-data-server==0.6.1
tensorboard-plugin-wit==1.8.0
tensorflow==2.5.0
tensorflow-estimator==2.5.0
termcolor==1.1.0
terminado==0.9.4
testpath @ file:///home/ktietz/src/ci/testpath_1611930608132/work
Theano==1.0.5
thinc==8.0.7
threadpoolctl==2.1.0
tokenizers==0.10.3
torch==1.8.1
torchaudio==0.8.1
torchvision==0.9.1
tornado @ file:///opt/concourse/worker/volumes/live/05341796-4198-4ded-4a9a-332fde3cdfd1/volume/tornado_1606942323372/work
tqdm==4.61.1
traitlets @ file:///home/ktietz/src/ci/traitlets_1611929699868/work
transformers==4.6.1
typer==0.3.2
typing-extensions==3.7.4.3
urllib3==1.26.5
wasabi==0.8.2
wcwidth @ file:///tmp/build/80754af9/wcwidth_1593447189090/work
webencodings==0.5.1
websocket-client==1.1.0
Werkzeug==2.0.1
widgetsnbextension==3.5.1
wrapt==1.12.1
XlsxWriter==1.4.5
zipp @ file:///tmp/build/80754af9/zipp_1615904174917/work
